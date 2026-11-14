import numpy as np
from PIL import Image

from alacv.imaging import build_pyramid, fit_to_input, open_image, save_image, to_rgb


def test_fit_to_input_square_landscape(image_factory):
    p = image_factory("a.jpg", 640, 480)
    img = open_image(p).image
    out = fit_to_input(img, 518)
    assert out.size == (518, 518)


def test_fit_to_input_portrait(image_factory):
    p = image_factory("b.png", 300, 600, fmt="PNG")
    img = open_image(p).image
    out = fit_to_input(img, 518)
    assert out.size == (518, 518)


def test_pyramid_sizes(image_factory):
    p = image_factory("c.jpg", 800, 600)
    img = open_image(p).image
    pyr = build_pyramid(img, [1, 2, 4, 8])
    assert sorted(pyr) == [1, 2, 4, 8]
    assert pyr[1].size == (800, 600)
    assert pyr[2].size == (400, 300)
    assert pyr[4].size == (200, 150)
    assert pyr[8].size == (100, 75)


def test_alpha_composited_on_white():
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[..., 3] = 0
    img = Image.fromarray(arr, mode="RGBA")
    out = to_rgb(img)
    assert out.mode == "RGB"
    assert np.asarray(out).mean() > 250


def test_save_webp_smaller_than_png(image_factory, tmp_path):
    p = image_factory("d.jpg", 512, 512)
    img = open_image(p).image
    n_webp = save_image(img, tmp_path / "x.webp", "webp", 80)
    n_png = save_image(img, tmp_path / "x.png", "png", 100)
    assert n_webp < n_png
    assert (tmp_path / "x.webp").exists()


def test_open_image_rejects_dir(tmp_path):
    try:
        open_image(tmp_path)
    except Exception as exc:
        assert isinstance(exc, (IsADirectoryError, PermissionError, OSError))
    else:
        raise AssertionError("目录不应被当作图像打开")
