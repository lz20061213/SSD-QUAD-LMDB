# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: anno.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
  name='anno.proto',
  package='',
  serialized_pb=_b('\n\nanno.proto\"\x81\x01\n\x05\x44\x61tum\x12\x10\n\x08\x63hannels\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\x0c\x12\r\n\x05label\x18\x05 \x01(\x05\x12\x12\n\nfloat_data\x18\x06 \x03(\x02\x12\x16\n\x07\x65ncoded\x18\x07 \x01(\x08:\x05\x66\x61lse\"\x8d\x01\n\nAnnotation\x12\n\n\x02x1\x18\x01 \x01(\x02\x12\n\n\x02y1\x18\x02 \x01(\x02\x12\n\n\x02x2\x18\x03 \x01(\x02\x12\n\n\x02y2\x18\x04 \x01(\x02\x12\n\n\x02x3\x18\x05 \x01(\x02\x12\n\n\x02y3\x18\x06 \x01(\x02\x12\n\n\x02x4\x18\x07 \x01(\x02\x12\n\n\x02y4\x18\x08 \x01(\x02\x12\x0c\n\x04name\x18\t \x01(\t\x12\x11\n\tdifficult\x18\n \x01(\x08\"Z\n\x0e\x41nnotatedDatum\x12\x15\n\x05\x64\x61tum\x18\x01 \x01(\x0b\x32\x06.Datum\x12\x10\n\x08\x66ilename\x18\x02 \x01(\t\x12\x1f\n\nannotation\x18\x03 \x03(\x0b\x32\x0b.Annotation')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_DATUM = _descriptor.Descriptor(
  name='Datum',
  full_name='Datum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='channels', full_name='Datum.channels', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height', full_name='Datum.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width', full_name='Datum.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='Datum.data', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='label', full_name='Datum.label', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='float_data', full_name='Datum.float_data', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='encoded', full_name='Datum.encoded', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15,
  serialized_end=144,
)


_ANNOTATION = _descriptor.Descriptor(
  name='Annotation',
  full_name='Annotation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x1', full_name='Annotation.x1', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y1', full_name='Annotation.y1', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='x2', full_name='Annotation.x2', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y2', full_name='Annotation.y2', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='x3', full_name='Annotation.x3', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y3', full_name='Annotation.y3', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='x4', full_name='Annotation.x4', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y4', full_name='Annotation.y4', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='name', full_name='Annotation.name', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='difficult', full_name='Annotation.difficult', index=9,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=147,
  serialized_end=288,
)


_ANNOTATEDDATUM = _descriptor.Descriptor(
  name='AnnotatedDatum',
  full_name='AnnotatedDatum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='datum', full_name='AnnotatedDatum.datum', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='filename', full_name='AnnotatedDatum.filename', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='annotation', full_name='AnnotatedDatum.annotation', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=290,
  serialized_end=380,
)

_ANNOTATEDDATUM.fields_by_name['datum'].message_type = _DATUM
_ANNOTATEDDATUM.fields_by_name['annotation'].message_type = _ANNOTATION
DESCRIPTOR.message_types_by_name['Datum'] = _DATUM
DESCRIPTOR.message_types_by_name['Annotation'] = _ANNOTATION
DESCRIPTOR.message_types_by_name['AnnotatedDatum'] = _ANNOTATEDDATUM

Datum = _reflection.GeneratedProtocolMessageType('Datum', (_message.Message,), dict(
  DESCRIPTOR = _DATUM,
  __module__ = 'anno_pb2'
  # @@protoc_insertion_point(class_scope:Datum)
  ))
_sym_db.RegisterMessage(Datum)

Annotation = _reflection.GeneratedProtocolMessageType('Annotation', (_message.Message,), dict(
  DESCRIPTOR = _ANNOTATION,
  __module__ = 'anno_pb2'
  # @@protoc_insertion_point(class_scope:Annotation)
  ))
_sym_db.RegisterMessage(Annotation)

AnnotatedDatum = _reflection.GeneratedProtocolMessageType('AnnotatedDatum', (_message.Message,), dict(
  DESCRIPTOR = _ANNOTATEDDATUM,
  __module__ = 'anno_pb2'
  # @@protoc_insertion_point(class_scope:AnnotatedDatum)
  ))
_sym_db.RegisterMessage(AnnotatedDatum)


# @@protoc_insertion_point(module_scope)
