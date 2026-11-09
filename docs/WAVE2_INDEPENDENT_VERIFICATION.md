# Wave 2 — عقد التحقق المستقل والآثار الخارجية

## 1. الغرض والنطاق

يحدد هذا المستند العقد التشغيلي الرسمي بين أربع طبقات مستقلة في KAI:

1. **Policy وApproval**: يقرران هل يُسمح بالتنفيذ.
2. **ToolGateway وProvider Effect**: يضمنان عدم تكرار الأثر الخارجي.
3. **Provider Adapter**: يرسل الطلب ويعيد نتيجة أولية.
4. **Independent Verification**: يثبت أن الحالة التي حفظها المزود تطابق الطلب المصرح به.

هذا العقد يخص عمليات الكتابة الخارجية، ويبدأ تغطيته التنفيذية الحالية بـ
`calendar.create_event`. ينطبق التصميم نفسه على أي أداة كتابة أخرى بعد إدخالها
إلى المسار المحكوم واختبارها بصورة مستقلة.

القاعدة المركزية هي:

```text
نجاح المزود لا يساوي نجاح التحقق.
تطبيق الأثر لا يساوي إثبات صحته.
```

## 2. المصطلحات الرسمية

### 2.1 Provider Effect

الأثر الخارجي هو التغيير الذي قد يكون حدث لدى المزود، مثل إنشاء حدث تقويم.
يُسجل في جدول `provider_effects` ويرتبط بمفتاح `idempotency_key` ثابت لا يجوز
تكراره.

### 2.2 Provider Result

هي النتيجة الأولية التي يعيدها adapter بعد الاتصال بالمزود. قد تحتوي على مرجع
مثل `event_id`، لكنها لا تثبت وحدها أن جميع الحقول تطابق الطلب.

### 2.3 Independent Verification

هو read-back مستقل من المزود، ثم مقارنة حتمية field-by-field مع الطلب المطبع.
بالنسبة إلى `calendar.create_event` تشمل المقارنة:

- `title` مقابل `summary`؛
- وقت البداية؛
- وقت النهاية؛
- المنطقة الزمنية؛
- مجموعة عناوين المشاركين بعد التطبيع؛
- وجود مرجع المزود `event_id` عند تحقق عقد read-after-create.

لا يعتمد التحقق على نص رسالة المزود، ولا على أن حالة HTTP هي `2xx` فقط.

## 3. invariants غير القابلة للتفاوض

### I-1 — لا نجاح دون تحقق مستقل

لا يجوز إنتاج:

```text
ToolOutcome.EXECUTION_SUCCEEDED
ok = true
verified = true
```

إلا إذا تحققت جميع الشروط التالية:

- التنفيذ وصل إلى المزود؛
- يوجد provider reference صالح؛
- نجح read-back؛
- تطابقت الحقول المطلوبة بالكامل؛
- أمكن تسجيل النتيجة الآمنة؛
- حالة الأثر أصبحت `APPLIED`.

### I-2 — فشل التحقق لا يعني عدم تطبيق الأثر

عندما يعيد المزود مرجعًا صالحًا، ثم يفشل read-back أو يكشف mismatch، يجب فصل
النتيجتين:

```text
provider_effect = APPLIED
ToolResult = REQUIRES_ATTENTION
verified = false
```

لا يجوز تحويل هذه الحالة إلى إعادة إرسال تلقائية؛ فقد يكون الحدث موجودًا بالفعل.

### I-3 — الغموض يمنع إعادة الإرسال

إذا انتهى الاتصال بمهلة أو خطأ لا يسمح بتحديد ما إذا كان المزود طبق الأثر،
تكون الحالة:

```text
provider_effect = AMBIGUOUS
```

ويجب توجيهها إلى reconciliation، لا إلى retry تلقائي.

### I-4 — reservation سابق يمنع executor

عند وجود سجل سابق بنفس مفتاح idempotency، سواء كان `RESERVED` أو `APPLIED` أو
`AMBIGUOUS`، يجب أن يتوقف Gateway قبل استدعاء executor ويعيد نتيجة attention.

### I-5 — لا أسرار في الدليل

لا يجوز أن تحتوي نتيجة التحقق أو `provider_effects.evidence` على:

- access token؛
- refresh token؛
- كلمات مرور؛
- نص رسالة بريد خام؛
- headers حساسة؛
- body كامل غير مطلوب للتدقيق.

يُحفظ فقط المرجع الخارجي والحقول الملخصة الضرورية لإثبات القرار.

## 4. مصفوفة الحالات والنتائج

| الحالة التشغيلية | Provider Effect | `ToolResult.outcome` | `ok` | `verified` | الإجراء التالي |
|---|---|---|---:|---:|---|
| حجز جديد قبل الاتصال | `RESERVED` | غير نهائي أثناء التنفيذ | — | — | يسمح بمحاولة العامل المالك فقط |
| انهيار العامل بعد commit وقبل الاتصال | `RESERVED` | لا توجد نتيجة موثوقة | — | — | reconciliation؛ لا redispatch تلقائي |
| المزود قبل الأثر وread-back مطابق بالكامل | `APPLIED` | `EXECUTION_SUCCEEDED` | true | true | إكمال المسار |
| المزود قبل الأثر لكن العنوان أو الوقت أو المشاركون مختلفون | `APPLIED` | `REQUIRES_ATTENTION` | false | false | مراجعة بشرية؛ لا retry |
| المزود أنشأ الأثر لكن read-back أعاد `404` أو تعذر قراءته | `APPLIED` | `REQUIRES_ATTENTION` | false | false | reconciliation؛ لا retry |
| timeout بعد dispatch أو نتيجة غير قابلة للتصنيف | `AMBIGUOUS` | `EXECUTION_FAILED` أو attention حسب طبقة الخطأ | false | false | reconciliation؛ لا retry |
| retry على `RESERVED` موجود | لا تغيير | `REQUIRES_ATTENTION` | false | false | لا يُستدعى executor |
| retry على `APPLIED` موجود | لا تغيير | `REQUIRES_ATTENTION` | false | false | لا يُستدعى executor |
| retry على `AMBIGUOUS` موجود | لا تغيير | `REQUIRES_ATTENTION` | false | false | لا يُستدعى executor |

> ملاحظة: `APPLIED` يصف تطبيق الأثر الخارجي، وليس نجاح التحقق. لذلك يمكن أن
> يجتمع `APPLIED` مع `REQUIRES_ATTENTION` بصورة صحيحة.

## 5. آلة حالات الأثر

الانتقالات الوحيدة المسموحة من خلال خدمة التسوية هي:

```text
RESERVED -> APPLIED
RESERVED -> AMBIGUOUS
```

لا توجد انتقالات تلقائية من الحالات النهائية، ولا يجوز تنفيذ:

```text
APPLIED -> RESERVED
APPLIED -> AMBIGUOUS
AMBIGUOUS -> RESERVED
AMBIGUOUS -> APPLIED
```

أما فشل التحقق بعد أن أعاد المزود `event_id` فيبقى `APPLIED`؛ لأن الأثر موجود،
وتُحفظ نتيجة التحقق في نتيجة Gateway والأدلة الآمنة، لا بتحريف حالة الأثر إلى
`AMBIGUOUS`.

## 6. عقد `calendar.create_event`

### قبل provider call

يجب على Gateway:

1. التحقق من الهوية والسياسة والموافقة؛
2. تطبيع والتحقق من arguments؛
3. اشتقاق `kai-<sha256>` idempotency key؛
4. حجز `provider_effect` بحالة `RESERVED`؛
5. تثبيت الحجز في معاملة مستقلة قبل الاتصال الخارجي؛
6. رفض أي حجز سابق قبل استدعاء adapter.

### بعد provider call

يجب على adapter/service:

1. الاحتفاظ بـ `event_id` إن أعاده المزود؛
2. تنفيذ read-back مستقل؛
3. تطبيق `verify_event` على كل الحقول؛
4. إعادة mismatches بصورة منظمة، لا كنص غير قابل للتحليل؛
5. عدم إدخال credentials أو body الخام في evidence.

### عند التسوية

- `execution.ok == true` يؤدي إلى `APPLIED` حتى لو كانت نتيجة التحقق غير ناجحة؛
- `verification.verified == true` فقط يسمح بـ `EXECUTION_SUCCEEDED`؛
- `verification.verified == false` ينتج `REQUIRES_ATTENTION`؛
- exception أو timeout أثناء provider exchange يؤدي إلى `AMBIGUOUS`؛
- أي retry لاحق يُحجب بواسطة المفتاح والحالة durable.

## 7. أدلة الاختبار الحالية

العقد مثبت باختبارات قابلة للتنفيذ في:

- [`test_calendar_independent_verification.py`](../tests/test_calendar_independent_verification.py): المقارنة الحتمية للعنوان والوقت والمنطقة الزمنية والمشاركين؛
- [`test_gateway_verification_effect_binding.py`](../tests/test_gateway_verification_effect_binding.py): ربط mismatch وread-back غير المتاح بـ `APPLIED` و`REQUIRES_ATTENTION` ومنع redispatch؛
- [`test_provider_effect_timeout_recovery.py`](../tests/test_provider_effect_timeout_recovery.py): تحويل timeout بعد dispatch إلى `AMBIGUOUS`؛
- [`test_provider_effect_crash_recovery.py`](../tests/test_provider_effect_crash_recovery.py): بقاء `RESERVED` بعد انهيار العامل قبل الاتصال؛
- [`test_provider_effect_concurrency.py`](../tests/test_provider_effect_concurrency.py): عاملان متزامنان ومفتاح واحد مع استدعاء executor واحد فقط.

## 8. معايير القبول الرسمية لـ Wave 2 — الجزء الأول

يُعد جزء Calendar Independent Verification مكتملًا عندما تكون الشروط التالية
خضراء في CI:

- mismatch في كل حقل جوهري لا ينتج نجاحًا؛
- read-back غير المتاح لا ينتج نجاحًا؛
- mismatch بعد إنشاء المزود يحفظ `APPLIED` ولا يعيد الإرسال؛
- timeout يحفظ `AMBIGUOUS` ولا يعيد الإرسال؛
- retry على أي حالة durable سابقة لا يصل إلى executor؛
- سباق عاملين على مفتاح واحد ينتج أثرًا واحدًا فقط؛
- لا تُسرّب الأسرار إلى evidence أو نتائج Gateway؛
- Ruff وMypy وmigration tests وnormal tests ناجحة.

## 9. ما لم يكتمل بعد

هذا العقد لا يعني أن reconciliation البشري أو استرداد `RESERVED` أصبحا مؤتمتين.
الخطوات اللاحقة في Wave 2 هي:

1. تعريف واجهة reconciliation آمنة للحالات `RESERVED` و`AMBIGUOUS`؛
2. تعميم عقد التحقق على Gmail بعد تثبيت مساره المحكوم في Gateway؛
3. إضافة production fail-closed gates وقياس قابلية التدقيق؛
4. نقل التنفيذ الطويل إلى worker مع lease وclaim ذري؛
5. ربط كل نتيجة تحقق مناسبة بالـ immutable evidence ledger.

حتى اكتمال هذه الخطوات، تبقى الحالات غير المؤكدة attention-required ولا يسمح النظام
بإعادة إرسال آلية قد تنتج أثرًا خارجيًا مكررًا.
