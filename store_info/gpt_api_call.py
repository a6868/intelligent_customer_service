import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def gpt_api_call(model_name,
                 user_text,
                 system_content='',
                 image_url='',
                 image_base64='',
                 frequency_penalty=0,
                 presence_penalty=-0.5,
                 max_tokens=4096,
                 temperature=0,
                 top_p=0.2,
                 n=1,
                 http_proxy=r"",
                 https_proxy=r"",
                 api_key_path=r""):
    '''
    model_name: 模型名称
    system_content: 系统提示词
    user_text: 用户输入
    image_url: 图片地址 
    return 返回completion不是返回内容
    '''
    import os
    import openai
    os.environ["http_proxy"] = http_proxy
    os.environ["https_proxy"] = https_proxy
    # 读取api key的txt文件
    with open(api_key_path, 'r') as f:
        text = f.read()
    openai.api_key = text
    if image_url:
      user_content = [{"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}]
    elif image_base64:
        base64_image = encode_image(image_base64)
        user_content = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "auto"
                }
            }
        ]
    else:
        user_content = user_text    
    if system_content:
      messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
        ]
    else:
      messages=[
        {"role": "user", "content": user_content}
      ]
    completion = openai.chat.completions.create(
        model = model_name,
        messages=messages,
        frequency_penalty=frequency_penalty,  
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n
    )
    return completion
