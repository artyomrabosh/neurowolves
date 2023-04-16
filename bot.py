import vk_api
import time
import main
import requests
import xml.etree.ElementTree as ET
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
# token https://oauth.vk.com/authorize?client_id=7938322&redirect_uri=https://oauth.vk.com/blank.html&display=page&scope=wall,manage,photos,docs&response_type=token&v=5.130

tree = ET.parse('bot_data.xml')
root = tree.getroot()
params = {
    "groupId": root.find('group').text,
    "intervalPost": int(root.find('interval').text),
    "v": root.find('v').text,
    "done": False,
    "lastpost": 1658991600,
    "token": root.find('token').text
}
VK = vk_api.VkApi(token = params["token"])

def load_big_gan():
    tf.disable_v2_behavior()
    tf.reset_default_graph()
    module_path = "GAN"
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
    return inputs, output

def get_token():
    pass

def fill_timetable():
    result = VK.method("wall.get", {
        "owner_id": params["groupId"],
        "domain": "neurovolchara",
        "filter": "postponed",
        "count": 100
    })
    posts = []
    for item in result["items"]:
        posts.append(item['date'])
    nearest_post = result["items"][0]
    dates_to_upload = []
    for i in range(99):
        date = nearest_post['date'] + i * 43200
        if date not in posts:
            dates_to_upload.append(date)
    return dates_to_upload



def publicPosts(inputs, output, date):
    upload = vk_api.VkUpload(VK)
    photo = upload.photo(
        "wolf.png",
        album_id=274136574,
        group_id=194187253
    )

    vk_photo_url = 'photo{}_{}'.format(
        photo[0]['owner_id'], photo[0]['id']
    )

    main.main(inputs, output)
    result = VK.method("wall.post", {
        "owner_id": params["groupId"],
        "message": None,
        "attachments": vk_photo_url,
        "copyright": "copyright",
        "v": params["v"],
        "publish_date": date
    })
    if result["post_id"]:
        print("Good post, id post - " + str(result["post_id"]))
    else:
        print("Error posting")


if __name__ == "__main__":
    inputs, output = load_big_gan()
    timetable = fill_timetable()
    print("Всего {} постов будет создано".format(len(timetable)))
    for date in timetable:
        publicPosts(inputs, output, date)
        time.sleep(params["intervalPost"])
    print("Gotcha!")

