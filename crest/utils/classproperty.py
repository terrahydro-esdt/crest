class classproperty:
    """ Allows class-level properties: https://stackoverflow.com/a/76301341/22210498
    
    Examples
    --------
    >>> class Test:
    ...     a = 2
    ...     @classproperty
    ...     def prop(cls):
    ...         return cls.a + 1
    >>> Test.prop
    3
    >>> Test().prop
    3

    """

    def __init__(self, func):           
        self.fget = func
    
    def __get__(self, instance, owner): 
        return self.fget(owner)