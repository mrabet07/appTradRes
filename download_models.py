from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

models = {
    ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
    ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
    ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
    ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
    ('en', 'ar'): 'Helsinki-NLP/opus-mt-en-ar',
    ('ar', 'en'): 'Helsinki-NLP/opus-mt-ar-en',
    ('ar', 'fr'): 'Helsinki-NLP/opus-mt-ar-fr',
    ('fr', 'ar'): 'Helsinki-NLP/opus-mt-fr-ar',
    ('ar', 'es'): 'Helsinki-NLP/opus-mt-ar-es',
    ('es', 'ar'): 'Helsinki-NLP/opus-mt-es-ar',
    ('es', 'fr'): 'Helsinki-NLP/opus-mt-es-fr',
    ('fr', 'es'): 'Helsinki-NLP/opus-mt-fr-es',
}

local_model_dir = './models'  # Dossier où les modèles seront stockés localement

for (src_lang, dest_lang), model_name in models.items():
    print(f"Downloading model for {src_lang} to {dest_lang}")
    AutoTokenizer.from_pretrained(model_name, cache_dir=local_model_dir)
    AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=local_model_dir)