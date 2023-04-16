import random
import re
import tensorflow.compat.v1 as tf
import os
import io

import numpy as np
import PIL.Image
from PIL import ImageDraw, ImageFont
from scipy.stats import truncnorm
import tensorflow_hub as hub

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main():
    with open("stervacut.txt", "r", encoding='utf-8') as f:
        txt = "".join(f.readlines())

    txt = txt.split("</s>\n")
    print(txt[:5])

    module_path = "GAN"
    tf.disable_v2_behavior()
    tf.reset_default_graph()
    print('Loading BigGAN module from:', module_path)
    module = hub.Module(module_path)
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in module.get_input_info_dict().items()}
    output = module(inputs)

    print()
    print('Inputs:\n', '\n'.join(
        '  {}: {}'.format(*kv) for kv in inputs.items()))
    print()
    print('Output:', output)

    input_z = inputs['z']
    input_y = inputs['y']
    input_trunc = inputs['truncation']

    dim_z = input_z.shape.as_list()[1]
    vocab_size = input_y.shape.as_list()[1]

    initializer = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initializer)
    np.random.seed(42)
    torch.manual_seed(42)

    OFFSET = 20
    FILL = (255, 255, 255, 255)
    SHADOW_FILL = (0, 0, 0, 255)
    dh = 50

    def draw_shadow(draw, x, y, text, font, shadowcolor):
        draw.text((x - 1, y), text, font=font, fill=shadowcolor)
        draw.text((x + 1, y), text, font=font, fill=shadowcolor)
        draw.text((x, y - 1), text, font=font, fill=shadowcolor)
        draw.text((x, y + 1), text, font=font, fill=shadowcolor)

        # thicker border
        draw.text((x - 1, y - 1), text, font=font, fill=shadowcolor)
        draw.text((x + 1, y - 1), text, font=font, fill=shadowcolor)
        draw.text((x - 1, y + 1), text, font=font, fill=shadowcolor)
        draw.text((x + 1, y + 1), text, font=font, fill=shadowcolor)

    def draw_text(image, txt):
        d = ImageDraw.Draw(image)
        fnt = ImageFont.truetype('arial.ttf', 40)
        w, h = image.size
        if len(txt) > 20:
            st = ""
            h_offset = 0
            splt = txt.split()
            for i in range(len(splt)):
                if len(st + " " + splt[i]) < 13 or splt[i] in ",.!&:;-":
                    st += " " + splt[i]
                    if i == len(splt) - 1:
                        draw_shadow(d, OFFSET, dh + h_offset, st, fnt, SHADOW_FILL)
                        d.text((OFFSET, dh + h_offset), st, font=fnt, fill=FILL)

                        break
                else:
                    st += " " + splt[i]
                    draw_shadow(d, OFFSET, dh + h_offset, st, fnt, SHADOW_FILL)
                    d.text((OFFSET, dh + h_offset), st, font=fnt, fill=FILL)

                    h_offset += 50
                    st = ""
        else:
            draw_shadow(d, OFFSET, dh, txt, fnt, SHADOW_FILL)
            d.text((OFFSET, dh), txt, font=fnt, fill=FILL)

    def truncated_z_sample(batch_size, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
        return truncation * values

    def one_hot(index, vocab_size=vocab_size):
        index = np.asarray(index)
        if len(index.shape) == 0:
            index = np.asarray([index])
        assert len(index.shape) == 1
        num = index.shape[0]
        output = np.zeros((num, vocab_size), dtype=np.float32)
        output[np.arange(num), index] = 1
        return output

    def one_hot_if_needed(label, vocab_size=vocab_size):
        label = np.asarray(label)
        if len(label.shape) <= 1:
            label = one_hot(label, vocab_size)
        assert len(label.shape) == 2
        return label

    def sample(sess, noise, label, truncation=1., batch_size=8,
               vocab_size=vocab_size):
        noise = np.asarray(noise)
        label = np.asarray(label)
        num = noise.shape[0]
        if len(label.shape) == 0:
            label = np.asarray([label] * num)
        if label.shape[0] != num:
            raise ValueError('Got # noise samples ({}) != # label samples ({})'
                             .format(noise.shape[0], label.shape[0]))
        label = one_hot_if_needed(label, vocab_size)
        ims = []
        for batch_start in range(0, num, batch_size):
            s = slice(batch_start, min(num, batch_start + batch_size))
            feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
            ims.append(sess.run(output, feed_dict=feed_dict))
        ims = np.concatenate(ims, axis=0)
        assert ims.shape[0] == num
        ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
        ims = np.uint8(ims)
        return ims

    def imgrid(imarray, cols=5, pad=1):
        if imarray.dtype != np.uint8:
            raise ValueError('imgrid input imarray must be uint8')
        pad = int(pad)
        assert pad >= 0
        cols = int(cols)
        assert cols >= 1
        N, H, W, C = imarray.shape
        rows = N // cols + int(N % cols != 0)
        batch_pad = rows * cols - N
        assert batch_pad >= 0
        post_pad = [batch_pad, pad, pad, 0]
        pad_arg = [[0, p] for p in post_pad]
        imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
        H += pad
        W += pad
        grid = (imarray
                .reshape(rows, cols, H, W, C)
                .transpose(0, 2, 1, 3, 4)
                .reshape(rows * H, cols * W, C))
        if pad:
            grid = grid[:-pad, :-pad]
        return grid

    def imshow(a, format='png', jpeg_fallback=True):
        a = np.asarray(a, dtype=np.uint8)
        data = io.BytesIO()
        img = PIL.Image.fromarray(a).resize((512, 512))
        draw_text(img, gen_citation())
        img.save("wolf.png", format)

    def load_tokenizer_and_model(model_name_or_path):
        return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path)

    def generate(
            model, tok, text,
            do_sample=True, max_length=50, repetition_penalty=5.0,
            top_k=5, top_p=0.95, temperature=1,
            num_beams=None,
            no_repeat_ngram_size=3
    ):
        input_ids = tok.encode(text, return_tensors="pt")
        out = model.generate(
            input_ids,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            top_k=top_k, top_p=top_p, temperature=temperature,
            num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
        )
        return list(map(tok.decode, out))

    """### RuGPT3Small"""

    def get_wolf():
        wolves_idx = [269, 270]
        trunc = random.choice(np.arange(0, 1, 0.1))
        noise = random.choice(range(100))
        z = truncated_z_sample(1, trunc, noise)
        y = random.choice(wolves_idx)

        ims = sample(sess, z, y, truncation=trunc)
        res = imgrid(ims, cols=min(1, 5))
        imshow(res)

    def gen_citation():
        tok, model = load_tokenizer_and_model("rugpt/wolves")

        abc = """
        йцукенгшщзхъфывапролдячсмитьбюжэЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ.,?!"№;%:()-1234567890 '
        """
        zatravka = random.choice(txt).replace("\n", " ")
        zatravka = " ".join(zatravka.split()[:4])
        generated = generate(model, tok, zatravka, num_beams=1)
        generated = "".join(list(filter(lambda x: x in abc, generated[0].split("</")[0])))

        if len(generated) - len(zatravka) < 5:
            generated = gen_citation()
        return generated

    get_wolf()
    return gen_citation()


if __name__ == '__main__':
    main()