# Final Project

## Ingest: завантаження документів у локальну базу знань

`ingest.py` будує або доповнює локальний Chroma-індекс для `knowledge_search`.
Після запуску агент шукає не напряму у файлах, а в індексі з каталогу
`INDEX_DIR`.

### Підтримувані формати

Файли потрібно покласти в каталог, заданий через `DATA_DIR`:

```env
DATA_DIR=data
```

Підтримуються:

- `.pdf`
- `.txt`
- `.json`
- `.yaml`
- `.yml`
- `.html`
- `.htm`
- `.doc`
- `.docx`

JSON і YAML перетворюються на читабельний текст перед індексацією. Якщо JSON
або YAML невалідний, файл буде завантажений як plain text.

### Запуск

З каталогу `final-project`:

```powershell
python ingest.py
```

Під час запуску `ingest.py`:

1. читає документи з `DATA_DIR`;
2. додає metadata і теги;
3. фільтрує документи, якщо задано `INGEST_TAG_FILTERS`;
4. розбиває документи на chunks;
5. створює embeddings через OpenAI;
6. записує chunks у Chroma.

### Куди записується індекс

```env
INDEX_DIR=index
CHROMA_COLLECTION=final-project_kb
```

Фізично база зберігається в:

```text
final-project/index
```

`CHROMA_COLLECTION` задає ім’я collection всередині Chroma. `ingest.py` пише
документи в цю collection, а `knowledge_search` читає з неї.

### Видаляти базу чи доповнювати

Поведінка задається параметром:

```env
INGEST_REBUILD_INDEX=true
```

Значення:

- `true` - видалити існуючий `INDEX_DIR` і побудувати базу заново;
- `false` - не видаляти базу, а додати нові chunks в існуючий Chroma index.

За замовчуванням використовується rebuild-режим, щоб база відповідала поточному
набору файлів і фільтрів.

### Фільтрація документів

Фільтр задається через:

```env
INGEST_TAG_FILTERS=issuer_match:nbu
```

Якщо значення непорожнє, в embeddings і Chroma потраплять тільки документи, які
мають хоча б один із указаних тегів.

Приклади:

```env
INGEST_TAG_FILTERS=issuer_match:nbu
```

Індексувати тільки документи, де знайдені NBU-ключові слова.

```env
INGEST_TAG_FILTERS=issuer_match:nbu,issuer_match:nssmc
```

Індексувати документи, де знайдені NBU або "Національна комісія з цінних паперів та фондового ринку" ключові слова.

```env
INGEST_TAG_FILTERS=
```

Вимкнути фільтр і індексувати всі підтримувані файли.

### Теги органів

Під час ingest кожен документ отримує metadata:

- `issuer` - основний визначений орган;
- `issuer_key` - машинний ключ органу;
- `file_type` - формат файлу;
- `tags` - список тегів через кому.

Приклад:

```text
issuer:nbu,format:json,issuer_match:nbu
```

`issuer:*` - основна класифікація документа.

`issuer_match:*` - документ містить ключові слова відповідного органу. Саме ці
теги зручно використовувати для ingest-фільтрів, бо документ може бути законом
Верховної Ради, але містити важливі згадки НБУ.

Поточні ключі:

- `issuer_match:verkhovna_rada`
- `issuer_match:nbu`
- `issuer_match:cabinet_ministers`
- `issuer_match:president`
- `issuer_match:minjust`
- `issuer_match:nssmc`
- `issuer_match:tax_service`
- `issuer_match:constitutional_court`
- `issuer_match:supreme_court`

### Важливі умови

- Для запуску потрібен `OPENAI_API_KEY`, бо embeddings створюються через OpenAI.
- Якщо OpenAI quota вичерпана, ingest впаде з `insufficient_quota`.
- Старі `.doc` файли найкраще читаються, якщо в системі є `antiword`, `catdoc`
  або LibreOffice `soffice`. Без них використовується fallback-витяг тексту.
- Якщо змінюєш `INGEST_TAG_FILTERS`, бажано запускати з
  `INGEST_REBUILD_INDEX=true`, щоб стара база не змішувалась із новим набором
  документів.
- Після успішного ingest `knowledge_search` працює з індексом, а не з файлами
  напряму.
