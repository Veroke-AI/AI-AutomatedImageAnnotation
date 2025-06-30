def is_valid_yolo_class(user_prompts, model):
    """
    Checks if user-provided prompts are in the YOLO model's known classes.
    
    Args:
        user_prompts (str or list of str): One or more class names to check.
        model (YOLO): An already-loaded YOLO model.

    Returns:
        valid (list): Prompts that match known classes.
        invalid (list): Prompts not found in the model's class list.
    """
    if isinstance(user_prompts, str):
        user_prompts = [user_prompts]
    
    known_classes = model.names.values()
    known_classes_lower = [cls.lower() for cls in known_classes]

    valid = []
    invalid = []

    for prompt in user_prompts:
        if prompt.lower() in known_classes_lower:
            valid.append(prompt)
        else:
            invalid.append(prompt)
    
    return valid, invalid
