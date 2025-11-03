python3 inference.py \
                    --model_path Qwen/Qwen-Image-Edit \
                    --image_path ../assets/samples/flower/000000000031.jpg \
                    --edit_prompt "Change the color of the rose from red to blue" \
                    --cond_b 1.0 \
                    --cond_delta 1.10 \
