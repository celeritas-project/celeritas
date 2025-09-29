#pragma once

#include <vector>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/io/ImportOpticalModel.hh"
#include "celeritas/optical/ImportedModelAdapter.hh"
#include "celeritas/optical/Model.hh"

#include "../MieData.hh"

namespace celeritas
{
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
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using HostRef = HostCRef<MieData>;
    using DeviceRef = DeviceCRef<MieData>;
    using SPConstCoreMaterials
        = std::shared_ptr<::celeritas::MaterialParams const>;
    struct Input
    {
        ImportModelClass model{ImportModelClass::size_};
        std::vector<ImportMie> data;
    };

    static ModelBuilder make_builder(SPConstImported imported, Input input);

    MieModel(ActionId id, SPConstImported imported, Input input);

    void build_mfps(OptMatId mat, MfpBuilder& build) const final;
    void step(CoreParams const&, CoreStateHost&) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;
    //! Access data on the host
    HostRef const& host_ref() const { return data_.host_ref(); }
    //! Access data on the device
    DeviceRef const& device_ref() const { return data_.device_ref(); }

  private:
    ImportedModelAdapter imported_;
    CollectionMirror<MieData> data_;
};

}  // namespace optical
}  // namespace celeritas
