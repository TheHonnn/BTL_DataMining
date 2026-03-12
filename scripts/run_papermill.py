import papermill as pm
import os

# Danh sách chạy theo đúng thứ tự Pipeline Khai phá dữ liệu
notebooks = [
     # 01
    "notebooks/02_preprocess_feature.ipynb",
    #03
    "notebooks/04_modeling.ipynb",
    "notebooks/04b_semi_supervised.ipynb",
    #05
]

def run_all():
    if not os.path.exists("outputs/reports"):
        os.makedirs("outputs/reports")
        
    for nb in notebooks:
        print(f">>> Đang thực thi tự động: {nb}")
        output_path = f"outputs/reports/{nb.split('/')[-1]}"
        try:
            pm.execute_notebook(nb, output_path)
        except Exception as e:
            print(f"Lỗi tại {nb}: {e}")

if __name__ == "__main__":
    run_all()
    print("Kết quả trong outputs/reports/")