import requests
import os

# # 设置要发送的音频文件路径
# audio_file_path = r"G:\音乐\无损\Centuries-Fall Out Boy.flac"
# file_name = os.path.basename(audio_file_path)  # 提取文件名
# spk_list = 'Lee'
#
# # 读取音频流文件
# with open(audio_file_path, 'rb') as f:
#     stream = f.read()
#
# # 定义目标URL
# url = "http://113.219.237.106:43811"
#
# # 准备请求数据
# data = {
#     "file_name": file_name,  # 作为普通参数发送文件名
#     "spk_list": spk_list,
#     "tran": -12
# }
# files = {
#     "file": stream  # 将音频流文件作为文件上传
# }
#
# # 发送POST请求
# response = requests.post(url, files=files, data=data)
#
# # 检查响应状态
# if response.status_code == 200:
#     # 将响应内容写入文件
#     with open('output.wav', 'wb') as f:
#         for chunk in response.iter_content(chunk_size=1024):
#             f.write(chunk)
#     print(f"音频文件已成功保存到: {data['file_name']}")
# else:
#     print(f"请求失败，状态码: {response.status_code}")

response = requests.get("https://frp-dad.top:21520/?file_name=mede_mede.flac&spk_list=aris", verify=False)
# response = requests.get("http://localhost:9960/?file_name=mede_mede.flac&spk_list=aris", verify=False)
if response.status_code == 200:
    # 将响应内容写入文件
    with open('output.wav', 'wb+') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print(f"音频文件已成功保存到: output.wav")
else:
    print(f"请求失败，状态码: {response.status_code}")
