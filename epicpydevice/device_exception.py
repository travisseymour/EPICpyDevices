from epiclibcpp.epiclib import raise_device_exception

raise_device_exception = raise_device_exception


class Device_exception(Exception):
    def __init__(self, message):
        super(Device_exception, self).__init__(message)
        raise_device_exception(message)  # TODO: WHY??


if __name__ == "__main__":
    raise Device_exception("This is a problem")
