def text_submit(chat, user_msg, arg_brain, arg_paras):
    chat = chat 
    user_msg = user_msg.strip()
    if not user_msg:
        chat.append((user_msg, "Please input text."))
        return chat, "", arg_brain, arg_paras
    
    brain = arg_brain.get("brain")
    say = brain.decide(user_msg, arg_paras)
    chat.append((user_msg, say))
    
    return chat, "", arg_brain, arg_paras

def file_upload(chat, file, arg_brain, arg_paras):
    print("Uploaded File Info:", file)
    chat = chat
    if not file:
        chat.append(('Upload an empty file!',"Please upload file."))
        return chat, arg_brain, arg_paras

    csv_path = file.name if hasattr(file, "name") else (file.get("name") if isinstance(file, dict) else str(file))

    brain = arg_brain.get("brain")
    say = brain.decide(csv_path,arg_paras)
    chat.append(('Upload file success!', say))
    return chat, arg_brain, arg_paras