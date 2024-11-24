import logging

class StreamingLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    Source: https://stackoverflow.com/a/36296215/22210498
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                line = line.strip()
                if line:
                    self.logger.log(self.log_level, line.strip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            line, self.linebuf = self.linebuf.strip(), ''
            if not (line.startswith('0%') or line.startswith('100%')):
                self.logger.log(self.log_level, line)
        