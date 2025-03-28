from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class NmrSchemaEntryPoint(SchemaPackageEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_nmr_schema.schema_packages.schema_package import m_package

        return m_package


nmr_schema = NmrSchemaEntryPoint(
    name='NmrSchemaPackage',
    description='Entry point configuration for NMR schema package.',
)
