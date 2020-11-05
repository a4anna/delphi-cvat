from delphi.attribute_provider import AttributeProvider


class ObjectProvider(object):

    def __init__(self, id: str, content: bytes, attributes: AttributeProvider, gt: bool):
        self.id = id
        self.content = content
        self.attributes = attributes
        self.gt = gt
