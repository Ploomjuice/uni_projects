from neomodel import (db, config, StructuredNode, StructuredRel, StringProperty,
                      IntegerProperty, UniqueIdProperty, RelationshipTo, FloatProperty)
config.DATABASE_URL = 'bolt://neo4j:neo4j@localhost:7687'


class Sample(StructuredNode):
    id = UniqueIdProperty()
    name = StringProperty(unique_index=True)
    length = FloatProperty()
    atk = FloatProperty()
    rel = FloatProperty()
    bass = FloatProperty()
    mids = FloatProperty()
    treble = FloatProperty()
    air = FloatProperty()
    group = IntegerProperty()
    similar = RelationshipTo('Sample', 'SIMILAR_TO')
    opposite = RelationshipTo('Sample', 'OPPOSITE_FROM')

class SimilarTo(StructuredRel):
    weight = FloatProperty()

class OppositeTo(StructuredRel):
    weight = FloatProperty()

def connect_similar(s1, s2, distance):
    connect = SimilarTo(s1, s2, weight=distance)
    connect.save()

def connect_different(s1, s2, distance):
    connect = OppositeTo(s1, s2, weight=distance)
    connect.save()

def exec_query(query):
    print(query)
    result = db.cypher_query(query)
    return result
