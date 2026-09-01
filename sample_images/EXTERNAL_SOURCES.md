# External smoke-test images

This folder contains a balanced local-only sanity-check set: 10 authentic-image
inputs (`external_real`) and 10 AI-generated images (`external_aigc`). The labels
are listed in `external_labels.csv`, where `0` means REAL and `1` means AIGC.

These 20 images are for qualitative smoke testing only. They are too few and too
selectively sourced to be used as a benchmark, to tune the model threshold, or to
support an accuracy claim. They must remain separate from SID training and holdout
data.

The image directories are ignored by Git. Before publicly redistributing any
asset, verify its current source-page licence and preserve all required attribution.

## AIGC sources

| Local file | Generator/source | Source and licence note |
| --- | --- | --- |
| `aigc_01_chatgpt_foxgirl.png` | ChatGPT 5.0 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:AI_generated_fox-girl_with_magic_mop_in_library.png), marked public domain on the source page |
| `aigc_02_chatgpt_alone.jpg` | ChatGPT | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Alone_AI-generated_image.jpg), marked public domain on the source page |
| `aigc_03_dalle_golden_lady.jpg` | DALL-E | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:A_golden_skin_lady_wearing_ornamental_heads_and_golden_ornaments_on_her_body_standing_in_a_lake.jpg), CC BY-SA 4.0; attribution: Encik Tekateki |
| `aigc_04_dalle_hawk.png` | DALL-E | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:A_hawk_sits_atop_a_puzzle_globe.png), marked public domain; review the source page's derived-prompt attribution note |
| `aigc_05_invoke_anime.png` | InvokeAI style-preset example | [InvokeAI sample assets](https://github.com/invoke-ai/InvokeAI/tree/main/invokeai/app/services/style_preset_images/default_style_preset_images); verify asset-level redistribution terms |
| `aigc_06_invoke_architecture.png` | InvokeAI style-preset example | Same source directory; verify asset-level redistribution terms |
| `aigc_07_invoke_character.png` | InvokeAI style-preset example | Same source directory; verify asset-level redistribution terms |
| `aigc_08_invoke_environment.png` | InvokeAI style-preset example | Same source directory; verify asset-level redistribution terms |
| `aigc_09_invoke_landscape.png` | InvokeAI style-preset example | Same source directory; verify asset-level redistribution terms |
| `aigc_10_invoke_product.png` | InvokeAI style-preset example | Same source directory; verify asset-level redistribution terms |

## REAL sources

The first eight files below are authentic input photographs distributed as
inpainting examples in the CompVis Stable Diffusion repository. The last two are
authentic input images used for super-resolution/upscaling examples. Repository
availability does not by itself establish permission to redistribute each photo,
so keep these local unless the original asset licence has been checked.

| Local file | Source |
| --- | --- |
| `real_01_flickr_6458524847.png` | [CompVis inpainting examples](https://github.com/CompVis/stable-diffusion/tree/main/data/inpainting_examples) |
| `real_02_flickr_8399166846.png` | Same source directory |
| `real_03_unsplash_alex_iby.png` | Same source directory |
| `real_04_bench.png` | Same source directory |
| `real_05_bertrand_gabioud.png` | Same source directory |
| `real_06_billow926.png` | Same source directory |
| `real_07_overture_creations.png` | Same source directory |
| `real_08_unsplash_photo.png` | Same source directory |
| `real_09_superresolution_input.jpg` | [CompVis super-resolution input](https://github.com/CompVis/stable-diffusion/blob/main/data/example_conditioning/superresolution/sample_0.jpg) |
| `real_10_upscaling_input.png` | [CompVis upscaling input](https://github.com/CompVis/stable-diffusion/blob/main/assets/stable-samples/img2img/upscaling-in.png) |
