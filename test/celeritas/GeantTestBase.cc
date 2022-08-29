//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GeantTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantTestBase.hh"

#include <string>

#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/process/BremsstrahlungProcess.hh"
#include "celeritas/em/process/ComptonProcess.hh"
#include "celeritas/em/process/EIonizationProcess.hh"
#include "celeritas/em/process/EPlusAnnihilationProcess.hh"
#include "celeritas/em/process/GammaConversionProcess.hh"
#include "celeritas/em/process/MultipleScatteringProcess.hh"
#include "celeritas/em/process/PhotoelectricProcess.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/random/RngParams.hh"

#include "celeritas_cmake_strings.h"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
namespace
{
//---------------------------------------------------------------------------//
std::string gdml_filename(const char* basename)
{
    return std::string(basename) + std::string(".gdml");
}

//---------------------------------------------------------------------------//
ImportData load_import_data(std::string filename)
{
    GeantPhysicsOptions options;
    options.em_bins_per_decade = 14;
    GeantImporter import(GeantSetup(filename, options));
    return import();
}

//---------------------------------------------------------------------------//
//! Test for equality of two C strings
bool cstring_equal(const char* lhs, const char* rhs)
{
    return std::strcmp(lhs, rhs) == 0;
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
//! Whether Geant4 dependencies match those on the CI build
bool GeantTestBase::is_ci_build()
{
    return cstring_equal(celeritas_rng, "XORWOW")
           && cstring_equal(celeritas_clhep_version, "2.4.4.0")
           && cstring_equal(celeritas_geant4_version, "10.7.2");
}

//---------------------------------------------------------------------------//
//! Whether Geant4 dependencies match those on Wildstyle
bool GeantTestBase::is_wildstyle_build()
{
    return cstring_equal(celeritas_rng, "XORWOW")
           && cstring_equal(celeritas_clhep_version, "2.4.5.1")
           && cstring_equal(celeritas_geant4_version, "10.7.3");
}

//---------------------------------------------------------------------------//
//! Whether Geant4 dependencies match those on the CI build
bool GeantTestBase::is_srj_build()
{
    return cstring_equal(celeritas_rng, "XORWOW")
           && cstring_equal(celeritas_clhep_version, "2.4.5.1")
           && cstring_equal(celeritas_geant4_version, "11.0.0");
}

//---------------------------------------------------------------------------//
// PROTECTED MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
auto GeantTestBase::build_material() -> SPConstMaterial
{
    return MaterialParams::from_import(this->imported_data());
}

//---------------------------------------------------------------------------//
auto GeantTestBase::build_geomaterial() -> SPConstGeoMaterial
{
    GeoMaterialParams::Input input;
    input.geometry       = this->geometry();
    input.materials      = this->material();
    const auto& imported = this->imported_data();

    input.volume_to_mat.resize(imported.volumes.size());
    input.volume_labels.resize(imported.volumes.size());
    for (auto volume_idx :
         range<VolumeId::size_type>(input.volume_to_mat.size()))
    {
        input.volume_to_mat[volume_idx]
            = MaterialId{imported.volumes[volume_idx].material_id};
        input.volume_labels[volume_idx]
            = Label::from_geant(imported.volumes[volume_idx].name);
    }
    return std::make_shared<GeoMaterialParams>(std::move(input));
}

//---------------------------------------------------------------------------//
auto GeantTestBase::build_particle() -> SPConstParticle
{
    return ParticleParams::from_import(this->imported_data());
}

//---------------------------------------------------------------------------//
auto GeantTestBase::build_cutoff() -> SPConstCutoff
{
    return CutoffParams::from_import(
        this->imported_data(), this->particle(), this->material());
}

//---------------------------------------------------------------------------//
auto GeantTestBase::build_physics() -> SPConstPhysics
{
    PhysicsParams::Input input;
    input.materials      = this->material();
    input.particles      = this->particle();
    input.options        = this->build_physics_options();
    input.action_manager = this->action_mgr().get();

    BremsstrahlungProcess::Options brem_options;
    brem_options.combined_model  = this->combined_brems();
    brem_options.enable_lpm      = true;
    brem_options.use_integral_xs = true;

    GammaConversionProcess::Options conv_options;
    conv_options.enable_lpm = true;

    EPlusAnnihilationProcess::Options epgg_options;
    epgg_options.use_integral_xs = true;

    EIonizationProcess::Options ioni_options;
    ioni_options.use_integral_xs = true;

    auto process_data
        = std::make_shared<ImportedProcesses>(this->imported_data().processes);
    input.processes.push_back(
        std::make_shared<ComptonProcess>(input.particles, process_data));
    input.processes.push_back(std::make_shared<PhotoelectricProcess>(
        input.particles, input.materials, process_data));
    input.processes.push_back(std::make_shared<GammaConversionProcess>(
        input.particles, process_data, conv_options));
    input.processes.push_back(std::make_shared<EPlusAnnihilationProcess>(
        input.particles, epgg_options));
    input.processes.push_back(std::make_shared<EIonizationProcess>(
        input.particles, process_data, ioni_options));
    input.processes.push_back(std::make_shared<BremsstrahlungProcess>(
        input.particles, input.materials, process_data, brem_options));
    if (this->enable_msc())
    {
        input.processes.push_back(std::make_shared<MultipleScatteringProcess>(
            input.particles, input.materials, process_data));
    }
    return std::make_shared<PhysicsParams>(std::move(input));
}

//---------------------------------------------------------------------------//
auto GeantTestBase::build_along_step() -> SPConstAction
{
    auto result
        = AlongStepGeneralLinearAction::from_params(*this->material(),
                                                    *this->particle(),
                                                    *this->physics(),
                                                    this->enable_fluctuation(),
                                                    this->action_mgr().get());
    CELER_ENSURE(result);
    CELER_ENSURE(result->has_fluct() == this->enable_fluctuation());
    CELER_ENSURE(result->has_msc() == this->enable_msc());
    CELER_ENSURE(&this->action_mgr()->action(result->action_id())
                 == result.get());
    return result;
}

//---------------------------------------------------------------------------//
auto GeantTestBase::build_physics_options() const -> PhysicsOptions
{
    PhysicsOptions options;
    options.secondary_stack_factor = this->secondary_stack_factor();
    return options;
}

//---------------------------------------------------------------------------//
// Lazily set up and load geant4
auto GeantTestBase::imported_data() const -> const ImportData&
{
    static struct
    {
        ImportData  imported;
        std::string geometry_basename;
    } i;
    std::string cur_basename = this->geometry_basename();
    if (i.geometry_basename != cur_basename)
    {
        CELER_VALIDATE(i.geometry_basename.empty(),
                       << "Geant4 currently crashes on second G4RunManager "
                          "instantiation (see issue #462)");
        i.geometry_basename = cur_basename;
        i.imported          = load_import_data(this->test_data_path(
            "celeritas", gdml_filename(cur_basename.c_str()).c_str()));
    }
    CELER_ENSURE(i.imported);
    return i.imported;
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, const PrintableBuildConf&)
{
    os << "RNG=\"" << celeritas_rng << "\", CLHEP=\""
       << celeritas_clhep_version << "\", Geant4=\""
       << celeritas_geant4_version << '"';
    return os;
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
