#pragma once

#include <vector>

#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/io/ImportOpticalModel.hh"
#include "celeritas/optical/ImportedModelAdapter.hh"
#include "celeritas/optical/Model.hh"

namespace celeritas
{
// class MaterialParams;
// struct ImportMie;
//  struct ImportOpticalRayleigh;
struct ImportData;
struct ImportMie;
namespace optical
{

class ImportedMaterials;
class MaterialParams;

//---------------------------------------------------------------------------//
class MieModel final : public Model
{
  public:
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    using SPConstImportedMaterials = std::shared_ptr<ImportedMaterials const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstCoreMaterials
        = std::shared_ptr<::celeritas::MaterialParams const>;
    struct Input
    {
        ImportModelClass model{ImportModelClass::size_};
        std::vector<ImportMie> data;
        //   SPConstMaterials materials;
        // SPConstCoreMaterials core_materials;
        // SPConstImportedMaterials imported_materials;
        // explicit operator bool() const
        //{
        //    return materials && core_materials && imported_materials;
        //}
    };

    static ModelBuilder make_builder(SPConstImported imported, Input input);

    MieModel(ActionId id, SPConstImported imported, Input input);

    void build_mfps(OptMatId mat, MfpBuilder& build) const final;
    void step(CoreParams const&, CoreStateHost&) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    ImportedModelAdapter imported_;
    Input input_;
    std::vector<ImportMie> mie_data_;
};

}  // namespace optical
}  // namespace celeritas
