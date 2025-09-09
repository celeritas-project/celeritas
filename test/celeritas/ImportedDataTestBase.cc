//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ImportedDataTestBase.cc
//---------------------------------------------------------------------------//
#include "ImportedDataTestBase.hh"

#include "geocel/SurfaceParams.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/ModelImporter.hh"
#include "celeritas/optical/PhysicsParams.hh"
#include "celeritas/optical/gen/CherenkovParams.hh"
#include "celeritas/optical/gen/ScintillationParams.hh"
#include "celeritas/optical/surface/SurfacePhysicsParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsOptions.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/track/SimParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_physics_options() const -> PhysicsOptions
{
    PhysicsOptions options;
    options.secondary_stack_factor = 3.0;
    return options;
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::select_optical_models() const -> std::vector<IMC>
{
    return {IMC::absorption, IMC::rayleigh, IMC::wls};
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_material() -> SPConstMaterial
{
    return MaterialParams::from_import(this->imported_data());
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_geomaterial() -> SPConstGeoMaterial
{
    // Access geometry first to build volume data
    auto geo = this->geometry();
    return GeoMaterialParams::from_import(
        this->imported_data(), geo, this->volume(), this->material());
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_particle() -> SPConstParticle
{
    return ParticleParams::from_import(this->imported_data());
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_cutoff() -> SPConstCutoff
{
    return CutoffParams::from_import(
        this->imported_data(), this->particle(), this->material());
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_sim() -> SPConstSim
{
    return std::make_shared<SimParams>([this] {
        return SimParams::Input::from_import(this->imported_data(),
                                             this->particle());
    }());
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_wentzel() -> SPConstWentzelOKVI
{
    return WentzelOKVIParams::from_import(
        this->imported_data(), this->material(), this->particle());
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_physics() -> SPConstPhysics
{
    using IPC = celeritas::ImportProcessClass;

    PhysicsParams::Input input;
    input.materials = this->material();
    input.particles = this->particle();
    input.options = this->build_physics_options();
    input.action_registry = this->action_reg().get();

    // Build processes
    auto const& imported = this->imported_data();
    ProcessBuilder build_process(imported, input.particles, input.materials);

    // Start with the ordering of processes from the original test harness
    std::vector<IPC> ipc{
        IPC::compton,
        IPC::photoelectric,
        IPC::conversion,
        IPC::annihilation,
        IPC::e_ioni,
        IPC::e_brems,
    };
    auto all_ipc = ProcessBuilder::get_all_process_classes(imported.processes);

    // Remove missing processes from `ipc` and found processes from `all_ipc`
    ipc.erase(std::remove_if(ipc.begin(),
                             ipc.end(),
                             [&all_ipc](ImportProcessClass i) {
                                 auto iter = all_ipc.find(i);
                                 if (iter == all_ipc.end())
                                     return true;
                                 all_ipc.erase(iter);
                                 return false;
                             }),
              ipc.end());
    // Add processes not in the original list to the end of the vector
    ipc.insert(ipc.end(), all_ipc.begin(), all_ipc.end());

    for (auto p : ipc)
    {
        input.processes.push_back(build_process(p));
        CELER_ASSERT(input.processes.back());
    }

    return std::make_shared<PhysicsParams>(std::move(input));
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_cherenkov() -> SPConstCherenkov
{
    return std::make_shared<CherenkovParams>(*this->optical_material());
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_optical_material() -> SPConstOpticalMaterial
{
    return optical::MaterialParams::from_import(
        this->imported_data(), *this->geomaterial(), *this->material());
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_scintillation() -> SPConstScintillation
{
    return ScintillationParams::from_import(this->imported_data(),
                                            this->particle());
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_optical_physics() -> SPConstOpticalPhysics
{
    using IMC = celeritas::optical::ImportModelClass;

    optical::PhysicsParams::Input input;
    input.materials = this->optical_material();
    input.action_registry = this->optical_action_reg().get();

    optical::ModelImporter importer(
        this->imported_data(), this->optical_material(), this->material());

    for (IMC imc : this->select_optical_models())
    {
        if (auto builder = importer(imc))
        {
            input.model_builders.push_back(*builder);
        }
    }

    return std::make_shared<optical::PhysicsParams>(std::move(input));
}

//---------------------------------------------------------------------------//
auto ImportedDataTestBase::build_optical_surface_physics()
    -> SPConstOpticalSurfacePhysics
{
    inp::SurfacePhysics input;

    // TODO: better input construction when we have actual data to import
    for (auto s : range(PhysSurfaceId{this->surface()->num_surfaces() + 1}))
    {
        input.materials.push_back(std::vector<OptMatId>{});
        input.roughness.polished.emplace(s, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(s, inp::FresnelReflection{});
        input.interaction.dielectric_dielectric.emplace(
            s, inp::ReflectionForm::from_spike());
    }

    return std::make_shared<optical::SurfacePhysicsParams>(
        this->optical_action_reg().get(), input);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
