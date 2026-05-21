import modal
import cloudvolume

image = (
    modal.Image.debian_slim()
    .uv_pip_install("cloud-volume")
)

app = modal.App("test-mip")

@app.function(image=image)
def test_mip():
    for mip in [0, 1, 2, 3, 4, 5]:
        try:
            vol = cloudvolume.CloudVolume('gs://h01-release/data/20210601/c3', use_https=True, mip=mip)
            print(f"MIP {mip}:")
            print(f"  Resolution: {vol.resolution}")
            print(f"  Bounds: {vol.bounds}")
        except Exception as e:
            print(f"MIP {mip} failed: {e}")

@app.local_entrypoint()
def main():
    test_mip.remote()
