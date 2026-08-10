from crest.base.BaseAbstract import BaseAbstract


class Writer(BaseAbstract):
    """ Base archive writing class.

    Notes
    -----
    Enables using inheriting classes as context managers that
    will automatically open and close the object. Also allows 
    objects to transparently access attributes defined within
    their underlying data schema and dask array; for example: 
    `ZarrWriter.numblocks == ZarrWriter.schema.data.numblocks`
    
    """
    
    def open(self):
        """ Opens the writer and initializes any containers """
        pass
    
    def close(self):
        """ Closes the writer and handles and pending data writes """
        pass

    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args, **kwargs):
        self.close()
    
    def __getattr__(self, attr):
        """ Enable direct access to schema / dask attributes """
        # Attempt to get attr in the order: self, schema, dask
        try: return self.__getattribute__(attr)
        except AttributeError as original_exc: 
            try: return getattr(self.schema, attr)
            except AttributeError: pass
            try: return getattr(self.schema.data, attr)
            except AttributeError: pass
            
            # Re-raise the original attribute error
            raise original_exc
                
