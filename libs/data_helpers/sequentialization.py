DICT_OPEN_TOKEN   = "[unused001]"
DICT_CLOSE_TOKEN  = "[unused002]"
LIST_OPEN_TOKEN   = "[unused003]"
LIST_CLOSE_TOKEN  = "[unused004]"
KEY_OPEN_TOKEN    = "[unused005]"
KEY_CLOSE_TOKEN   = "[unused006]"
VALUE_OPEN_TOKEN  = "[unused007]"
VALUE_CLOSE_TOKEN = "[unused008]"

mapping = {
    DICT_OPEN_TOKEN: "<DICT>",
    DICT_CLOSE_TOKEN: "</DICT>",
    LIST_OPEN_TOKEN: "<LIST>",
    LIST_CLOSE_TOKEN: "</LIST>",
    KEY_OPEN_TOKEN: "<KEY>",
    KEY_CLOSE_TOKEN: "</KEY>",
    VALUE_OPEN_TOKEN: "<VALUE>",
    VALUE_CLOSE_TOKEN: "</VALUE>"
}

def transform_json(item, readable: bool = False):
    if not isinstance(item, (list, dict)):
        return " " + item

    dict_open_token   = DICT_OPEN_TOKEN
    dict_close_token  = DICT_CLOSE_TOKEN
    list_open_token   = LIST_OPEN_TOKEN
    list_close_token  = LIST_CLOSE_TOKEN
    key_open_token    = KEY_OPEN_TOKEN
    key_close_token   = KEY_CLOSE_TOKEN
    value_open_token  = VALUE_OPEN_TOKEN
    value_close_token = VALUE_CLOSE_TOKEN

    if readable:
        dict_open_token   = mapping[dict_open_token]
        dict_close_token  = mapping[dict_close_token]
        list_open_token   = mapping[list_open_token]
        list_close_token  = mapping[list_close_token]
        key_open_token    = mapping[key_open_token]
        key_close_token   = mapping[key_close_token]
        value_open_token  = mapping[value_open_token]
        value_close_token = mapping[value_close_token]

    if isinstance(item, dict):
        template = "{}{{}}{}"
        out = template.format(dict_open_token, dict_close_token)
        inner = ""
        for k, v in item.items():
            inner += (
                "{}{}{}".format(key_open_token, transform_json(k, readable), key_close_token) +
                "{}{}{}".format(value_open_token, transform_json(v, readable), value_close_token)
            )
        return out.format(inner)

    # isinstance(item, list)
    template = "{}{{}}{}"
    out = template.format(list_open_token, list_close_token)
    inner = ""
    for r in item:
        inner += transform_json(r, readable)
    return out.format(inner)
