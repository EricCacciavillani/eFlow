def enum(**enums):
    """Allows for constant like variables.
    """
    return type('Enum', (), enums)