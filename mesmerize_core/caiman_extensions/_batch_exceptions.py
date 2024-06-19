class BatchItemNotRunError(Exception):
    pass


class BatchItemUnsuccessfulError(Exception):
    pass


class WrongAlgorithmExtensionError(Exception):
    pass


class DependencyError(Exception):
    pass


class PreventOverwriteError(IndexError):
    """  
    Error thrown when trying to write to an existing batch file with a potential risk of removing existing rows.  
    """  
    pass