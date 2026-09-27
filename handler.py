import runpod
import base64
import trimesh
import pymeshfix
from rembg import remove
from PIL import Image
import io
import os
import torch
from TripoSR import TripoSR  # كيجي من requirements

# نحملو الموديل مرة وحدة فاللول
print("... كاين تحميل الموديل TripoSR ...")
model = TripoSR.from_pretrained("stabilityai/TripoSR", device="cuda:0")
print("الموديل واجد!")

def handler(job):
    try:
        # 1. ناخدو الصورة من الزبون (جاية base64)
        image_base64 = job['input']['image_base64']
        image_bytes = base64.b64decode(image_base64)
        
        # 2. نقطعو الخلفية
        print("... قطع الخلفية ...")
        img_no_bg = remove(image_bytes)
        image = Image.open(io.BytesIO(img_no_bg)).convert("RGB")

        # 3. نحولو الصورة لـ 3D
        print("... تحويل لـ 3D ...")
        with torch.no_grad():
            mesh = model.predict(image) # كيخرج mesh

        # 4. نصلحو الثقوب باش يطبع
        print("... إصلاح الـ Mesh ...")
        vertices = mesh.vertices
        faces = mesh.faces
        
        # هاد المكتبة كتسد الثقوب
        meshfix = pymeshfix.MeshFix(vertices, faces)
        meshfix.repair()
        v, f = meshfix.v, meshfix.f

        # 5. نعطيوه سمك ونصدروه STL
        fixed_mesh = trimesh.Trimesh(vertices=v, faces=f)
        # نتأكدو واش Water-tight
        if not fixed_mesh.is_watertight:
            print("ماشي Water-tight، كنحاول نصلحو...")
            fixed_mesh.fill_holes()

        # نحفظوه
        output_path = "/tmp/output.stl"
        fixed_mesh.export(output_path)

        # 6. نرجعوه للزبون كـ base64
        with open(output_path, "rb") as f:
            stl_data = base64.b64encode(f.read()).decode('utf-8')

        return {
            "status": "success",
            "stl_base64": stl_data,
            "message": "تم بنجاح! الـ STL واجد للطباعة"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# هادي ضرورية باش RunPod يخدم
runpod.serverless.start({"handler": handler})