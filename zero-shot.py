import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
from transformers import CLIPProcessor, CLIPModel
import threading
import requests
import os

#بخش مدل

#بارگذاری مستقیم مدل
Model_Name = "openai/clip-vit-large-patch14"

Promts_Per_Class = {
    "cat": [
        "a photo of a cat.",
        "a picture of a cat.",
        "an image of a cat, a feline.",
        "a cat sitting on a couch.",
        "an animal that meows",
        "a close-up of a cat's face."
    ],
    "girl and fireworks": [
        "a photo of a girl.",
        "a picture of a girl.",
        "an image of a girl, japanese.",
        "a girl watching fireworks.",
        "a photo of fireworks.",
        "a picture of fireworks",
        "a young woman enjoying fireworks"
    ],
    "light souls": [
        "a photo of lighty souls.",
        "a picture of bright spirit glowing.",
        "a picture of society of souls.",
        "an image of plants, a jungle.",
        "a picture of butterflies glowing."
    ],
    "snowman": [
        "a photo of a little snowman.",
        "a picture of a snowman with a cute smile.",
        "an image of little lightss.",
        "a picture of a snowy park."
    ],
        "harbor": [
        "a photo of a harbor city.",
        "a picture of skyscrappers.",
        "a picture of beautiful sunshine.",
        "an image of sea."
    ],
        "Snowy Christmas Town": [
        "a photo of Christmas Vibe.",
        "a picture of a Snowy Christmas Town.",
        "a picture of christmas trees.",
        "an image of kids walking in snow, a snowy town.",
        "a picture of homes design for christmas."
    ]
}

def classify_image_with_clip(model, processor, image: Image.Image):
    class_names = list(Promts_Per_Class.keys())
    all_prompts = [prompt for prompts in Promts_Per_Class.values() for prompt in prompts]
    inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image.squeeze(0)
    class_logits = []
    prompt_counter = 0
    for class_name in class_names:
        num_prompts = len(Promts_Per_Class[class_name])
        class_specific_logits = logits_per_image[prompt_counter : prompt_counter + num_prompts]
        mean_logit = class_specific_logits.mean()
        class_logits.append(mean_logit)
        prompt_counter += num_prompts
    final_logits = torch.tensor(class_logits)
    final_probs = final_logits.softmax(dim=-1)
    results = {class_names[i]: final_probs[i].item() * 100 for i in range(len(class_names))}
    return results

# -------------------------------------------------------------------
# بخش رابط کاربری Tkinter

class ClipApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("طبقه‌بندی تصویر با CLIP")
        self.geometry("600x700")

        self.model = None
        self.processor = None
        self.image_path = None

        style = ttk.Style(self)
        style.configure("TButton", font=("Arial", 12), padding=10)
        style.configure("TLabel", font=("Arial", 12))
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        style.configure("Result.TLabel", font=("Arial", 16, "bold"), foreground="blue")

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        self.btn_select = ttk.Button(main_frame, text="انتخاب تصویر از پوشه 'images'", command=self.select_image, state="disabled")
        self.btn_select.pack(pady=10)

        self.lbl_image = ttk.Label(main_frame)
        self.lbl_image.pack(pady=10)

        self.status_label = ttk.Label(main_frame, text="درحال بارگذاری مدل، لطفا صبر کنید...", anchor="center")
        self.status_label.pack(fill="x", pady=5)
        
        results_frame = ttk.LabelFrame(main_frame, text="نتایج طبقه‌بندی", padding="10")
        results_frame.pack(fill="x", expand=True, pady=10)
        
        self.result_labels = {}
        for class_name in Promts_Per_Class.keys():
            frame = ttk.Frame(results_frame)
            frame.pack(fill="x", pady=2)
            label = ttk.Label(frame, text=f"{class_name}: -- %")
            label.pack(side="right")
            self.result_labels[class_name] = label
        
        ttk.Separator(main_frame).pack(fill='x', pady=10)
        self.final_prediction_label = ttk.Label(main_frame, text="پیش‌بینی نهایی: --", style="Result.TLabel", anchor="center")
        self.final_prediction_label.pack(pady=10)

        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            self.model = CLIPModel.from_pretrained(Model_Name)
            self.processor = CLIPProcessor.from_pretrained(Model_Name)
            self.status_label.config(text="مدل بارگذاری شد. لطفا یک تصویر انتخاب کنید.")
            self.btn_select.config(state="normal")
        except (requests.exceptions.RequestException, OSError) as e:
            messagebox.showerror("خطای شبکه", f"امکان بارگذاری مدل وجود ندارد. لطفا اتصال اینترنت خود را بررسی کنید.\n{e}")
            self.status_label.config(text="خطا در بارگذاری مدل!")

    def select_image(self):
        # مسیر پوشه تصاویر
        images_dir = os.path.join(os.getcwd(), "images")

        # اگر پوشه وجود نداشت، آن را ایجاد می‌کند تا از خطا جلوگیری شود
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            messagebox.showinfo("اطلاع", f"پوشه 'images' ایجاد شد. لطفا عکس‌های خود را در آن قرار دهید.")

        file_path = filedialog.askopenfilename(
            title="یک تصویر از پوشه 'images' انتخاب کنید",
            initialdir=images_dir, # <-- مسیر پیش‌فرض برای باز شدن پنجره
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        self.image_path = file_path
        self.display_image(file_path)
        self.clear_results()
        self.status_label.config(text="درحال پردازش تصویر...")
        threading.Thread(target=self.run_classification, daemon=True).start()

    def run_classification(self):
        try:
            image = Image.open(self.image_path).convert("RGB")
            results = classify_image_with_clip(self.model, self.processor, image)
            self.after(0, self.update_ui_with_results, results)
        except Exception as e:
            self.after(0, lambda: self.status_label.config(text=f"خطا در پردازش: {e}"))

    def display_image(self, path):
        img = Image.open(path)
        img.thumbnail((300, 300))
        self.photo_tk = ImageTk.PhotoImage(img)
        self.lbl_image.config(image=self.photo_tk)

    def update_ui_with_results(self, results):
        best_class = max(results, key=results.get)
        for class_name, prob in results.items():
            self.result_labels[class_name].config(text=f"{class_name}: {prob:.2f}%")
        self.final_prediction_label.config(text=f"پیش‌بینی نهایی: {best_class}")
        self.status_label.config(text="طبقه‌بندی با موفقیت انجام شد!")
        
    def clear_results(self):
        for label in self.result_labels.values():
            label.config(text=f"{label.cget('text').split(':')[0]}: -- %")
        self.final_prediction_label.config(text="پیش‌بینی نهایی: --")

if __name__ == "__main__":
    app = ClipApp()
    app.mainloop()