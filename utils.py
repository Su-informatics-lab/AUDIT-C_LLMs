

CSV_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_CX_04052024.csv'
DATASET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_CX_04052024'
MODEL_NAME = 'google/flan-t5-base'
PROJECT_NAME = 'AUDIT-C_LLMs'
SEED = 6179

MAX_LENGTH = 256
MAX_OUTPUT_LENGTH = 4
HEAD = ("### Score the user's Alcohol Use Disorders Identification Test (AUDIT-C) "
        "from 0 to 12 based on the provided demographics and comorbidity data:\n")
TAIL = "\n### AUDIT-C Score:"
