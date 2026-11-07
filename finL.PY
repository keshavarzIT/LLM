from transformers import LlamaTokenizer, LlamaForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np

# بارگذاری مدل
model_path = r"C:\chatbot\persian_llama_7B_merged"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

# مدل امبدینگ
embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# متن
context_text = """
دریاها گرمای خورشید را جذب و ذخیره کرده و در تعدیل آب و هوای زمین نقش مهمی دارند.
فیتوپلانکتون‌ها و جلبک‌های دریایی، مسئول تولید بخش بزرگی از اکسیژن مورد نیاز جهان هستند.
اقیانوس‌ها بیش از ۱۴ درصد پروتئین مورد نیاز بشر را تامین می‌کنند.
دریاها مسیرهای اصلی تجارت و حمل‌ونقل دریایی برای انسان‌ها بوده‌اند.
دریاها زیستگاه گونه‌های بی‌شمار جانوری و گیاهی، از ماهیان رنگارنگ گرفته تا صخره‌های مرجانی و کوسه‌های متنوع، هستند.
بخش بزرگی از کف اقیانوس‌ها هنوز ناشناخته باقی مانده است و پژوهشگران تخمین می‌زنند که تنها درصد بسیار کوچکی از آن کشف شده است.
فعالیت‌های آتشفشانی عظیم و رشته‌کوه‌های بلند در زیر آب، بخشی از شگفتی‌های دنیای زیر دریا هستند.
دریا همواره الهام‌بخش هنرمندان، نویسندگان و شاعران بوده و نمادی از آرامش، عظمت و اسرارآمیز بودن است.
از دوران پیشاتاریخ، انسان‌ها در حال دریانوردی و اکتشاف دریاها بوده‌اند که مطالعه علمی آن‌ها به قرن ۱۹ میلادی برمی‌گردد.
دریا، پهنه‌ای وسیع از آب است که بیش از ۷۰ درصد سطح زمین را پوشانده و نقش حیاتی در چرخه آب و هوای زمین ایفا می‌کند.
آب دریا طعم شوری دارد.
"""

# آماده‌سازی جملات
sentences = [s.strip() for s in context_text.split('\n') if s.strip()]
embeddings = embed_model.encode(sentences, convert_to_numpy=True)

# ساخت FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# سوال‌ها
questions = [

    "آب دریا چه طعمی دارد؟",
    "مطالعه علمی دریا به چه قرنی بر میگردد؟",
    "آیا آب دریا شیرین است؟",
    "ارتفاع بلندترین کوه دریا چقدر است؟",
    "چه چیزهایی مسئول تولید اکسیژن جهان هستند؟"
]

# ذخیره پاسخ‌ها
with open("answers_rag.txt", "w", encoding="utf-8") as f:
    for q in questions:
        q_emb = embed_model.encode([q], convert_to_numpy=True)
        D, I = index.search(q_emb, k=1)  
        retrieved_text = sentences[I[0][0]]

        prompt = f"فقط بر اساس متن زیر جواب بده و از هیچ چیز دیگر استفاده نکن. اگر پاسخ در متن نیست، بگو 'اطلاعاتی دراین خصوص در متن مشاهده نشد'.\nمتن: {retrieved_text}\nسوال: {q}\nپاسخ:"

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.001,
            top_p=1.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        f.write(f"سوال: {q}\nپاسخ: {response}\n\n")

print("the answers has been written")
