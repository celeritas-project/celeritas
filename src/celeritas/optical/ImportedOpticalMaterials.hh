#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A registry for imported optical material data.
 */
class ImportedOpticalMaterials
{
  public:
    //! Construct with imported data
    static std::shared_ptr<ImportedOpticalMaterials> from_import(ImportData const& data);

    //! Construct with imported tables
    explicit ImportedOpticalMaterials(std::vector<ImportOpticalMaterial> io);

    //! Get the optical material properties for the given ID
    inline ImportOpticalMaterial const& get(OpticalMaterialId id) const;

    //! Number of imported optical materials
    inline OpticalMaterialId::size_type size() const;

  private:
    std::vector<ImportOpticalMaterial> materials_;
};


ImportOpticalMaterial const& ImportedOpticalMaterials::get(OpticalMaterialId id) const
{
    CELER_EXPECT(id && id < this->size());
    return materials_[id.get()];
}

OpticalMaterialId::size_type ImportedOpticalMaterials::size() const
{
    return materials_.size();
}


}
