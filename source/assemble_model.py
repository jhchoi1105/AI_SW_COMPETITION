# -*- coding: utf-8 -*-
"""[4단계] 최종 제출물(g4_c15) 조립.

입력 (앞 단계 산출물):
  ./work/lgbm/{text_model.pkl, lgbm_final.txt, lgbm_classes.json}
  ./work/tf_s7_trimmed, tf_s77_trimmed, tf_s99_trimmed, tf_s123_trimmed
  ./final_params.json  (앙상블 하이퍼파라미터 — Private Score 기록본과 동일)
  ./final_inference/{script.py, requirements.txt}  (추론 코드)

산출물:
  ./final_submission/  (model/ + script.py + requirements.txt)
  ./final_submission.zip

실행:  python assemble_model.py
"""
import os
import shutil
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

DST = "./final_submission"
TF_MAP = {  # 제출물 내 폴더명 ← 학습·트림 산출물
    "tf_s77": "./work/tf_s77_trimmed",
    "tf_s7": "./work/tf_s7_trimmed",
    "tf_s99": "./work/tf_s99_trimmed",
    "tf_s123": "./work/tf_s123_trimmed",
}


def main():
    shutil.rmtree(DST, ignore_errors=True)
    os.makedirs(f"{DST}/model", exist_ok=True)

    for f in ("text_model.pkl", "lgbm_final.txt", "lgbm_classes.json"):
        shutil.copy(f"./work/lgbm/{f}", f"{DST}/model/{f}")
    shutil.copy("./final_params.json", f"{DST}/model/params.json")
    for name, src in TF_MAP.items():
        shutil.copytree(src, f"{DST}/model/{name}")
    shutil.copy("./final_inference/script.py", f"{DST}/script.py")
    shutil.copy("./final_inference/requirements.txt", f"{DST}/requirements.txt")

    out_zip = "./final_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(DST):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                fp = os.path.join(root, f)
                z.write(fp, os.path.relpath(fp, DST))
    print(f"조립 완료: {DST}  /  {out_zip} "
          f"({os.path.getsize(out_zip)/2**20:.0f}MiB)", flush=True)


if __name__ == "__main__":
    main()
