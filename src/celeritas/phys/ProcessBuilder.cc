//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ProcessBuilder.cc
//---------------------------------------------------------------------------//
#include "ProcessBuilder.hh"

#include <set>
#include <unordered_map>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/em/process/BremsstrahlungProcess.hh"
#include "celeritas/em/process/ComptonProcess.hh"
#include "celeritas/em/process/CoulombScatteringProcess.hh"
#include "celeritas/em/process/EIonizationProcess.hh"
#include "celeritas/em/process/EPlusAnnihilationProcess.hh"
#include "celeritas/em/process/GammaConversionProcess.hh"
#include "celeritas/em/process/MuBremsstrahlungProcess.hh"
#include "celeritas/em/process/MuIonizationProcess.hh"
#include "celeritas/em/process/MuPairProductionProcess.hh"
#include "celeritas/em/process/PhotoelectricProcess.hh"
#include "celeritas/em/process/RayleighProcess.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/NeutronXsReader.hh"
#include "celeritas/neutron/process/NeutronElasticProcess.hh"

#include "ImportedProcessAdapter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get an ordered set of all available processes.
 */
auto ProcessBuilder::get_all_process_classes(
    std::vector<ImportProcess> const& processes) -> std::set<IPC>
{
    std::set<ImportProcessClass> result;
    for (auto const& p : processes)
    {
        result.insert(p.process_class);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct imported process data.
 *
 * \pre The import data must have already been converted to the native unit
 * system.
 */
ProcessBuilder::ProcessBuilder(ImportData const& data,
                               SPConstParticle particle,
                               SPConstMaterial material,
                               UserBuildMap user_build)
    : data_{data}
    , input_{std::move(material), std::move(particle), nullptr}
    , user_build_map_(std::move(user_build))
    , enable_lpm_(data.em_params.lpm)
{
    CELER_EXPECT(input_.material);
    CELER_EXPECT(input_.particle);
    CELER_EXPECT(std::string(data.units) == units::NativeTraits::label());

    input_.imported = std::make_shared<ImportedProcesses>(data.processes);
}

//---------------------------------------------------------------------------//
/*!
 * Construct without custom user builders.
 */
ProcessBuilder::ProcessBuilder(ImportData const& data,
                               SPConstParticle particle,
                               SPConstMaterial material)
    : ProcessBuilder(
          data, std::move(particle), std::move(material), UserBuildMap{})
{
}

//---------------------------------------------------------------------------//
//! Default destructor
ProcessBuilder::~ProcessBuilder() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct a \c Process from a given process class.
 *
 * This may return a null process (with a warning) if the user specifically
 * requests that the process be omitted.
 */
auto ProcessBuilder::operator()(IPC ipc) -> SPProcess
{
    // First, look for user-supplied processes
    {
        auto user_iter = user_build_map_.find(ipc);
        if (user_iter != user_build_map_.end())
        {
            return user_iter->second(input_);
        }
    }

    using BuilderMemFn = SPProcess (ProcessBuilder::*)();
    static std::unordered_map<IPC, BuilderMemFn> const builtin_build{
        {IPC::annihilation, &ProcessBuilder::build_annihilation},
        {IPC::compton, &ProcessBuilder::build_compton},
        {IPC::conversion, &ProcessBuilder::build_conversion},
        {IPC::coulomb_scat, &ProcessBuilder::build_coulomb},
        {IPC::e_brems, &ProcessBuilder::build_ebrems},
        {IPC::e_ioni, &ProcessBuilder::build_eioni},
        {IPC::mu_brems, &ProcessBuilder::build_mubrems},
        {IPC::mu_ioni, &ProcessBuilder::build_muioni},
        {IPC::mu_pair_prod, &ProcessBuilder::build_mupairprod},
        {IPC::neutron_elastic, &ProcessBuilder::build_neutron_elastic},
        {IPC::photoelectric, &ProcessBuilder::build_photoelectric},
        {IPC::rayleigh, &ProcessBuilder::build_rayleigh},
    };

    // Next, try built-in processes
    {
        auto iter = builtin_build.find(ipc);
        CELER_VALIDATE(iter != builtin_build.end(),
                       << "cannot build unsupported EM process '" << ipc
                       << "'");

        BuilderMemFn build_impl{iter->second};
        auto result = (this->*build_impl)();
        CELER_ENSURE(result);
        return result;
    }
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_eioni() -> SPProcess
{
    return std::make_shared<EIonizationProcess>(this->particle(),
                                                this->imported());
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_ebrems() -> SPProcess
{
    CELER_EXPECT(data_.seltzer_berger);
    BremsstrahlungProcess::Options options;
    options.enable_lpm = enable_lpm_;

    return std::make_shared<BremsstrahlungProcess>(this->particle(),
                                                   this->material(),
                                                   this->imported(),
                                                   data_.seltzer_berger,
                                                   options);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_neutron_elastic() -> SPProcess
{
    // FIXME: read neutron data at import
    return std::make_shared<NeutronElasticProcess>(
        this->particle(), this->material(), NeutronXsReader{NeutronXsType::el});
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_photoelectric() -> SPProcess
{
    CELER_EXPECT(data_.livermore_photo);
    return std::make_shared<PhotoelectricProcess>(this->particle(),
                                                  this->material(),
                                                  this->imported(),
                                                  data_.livermore_photo);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_compton() -> SPProcess
{
    return std::make_shared<ComptonProcess>(this->particle(), this->imported());
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_conversion() -> SPProcess
{
    GammaConversionProcess::Options options;
    options.enable_lpm = enable_lpm_;

    return std::make_shared<GammaConversionProcess>(
        this->particle(), this->imported(), options);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_rayleigh() -> SPProcess
{
    return std::make_shared<RayleighProcess>(
        this->particle(), this->material(), this->imported());
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_annihilation() -> SPProcess
{
    return std::make_shared<EPlusAnnihilationProcess>(this->particle(),
                                                      this->imported());
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_coulomb() -> SPProcess
{
    return std::make_shared<CoulombScatteringProcess>(
        this->particle(), this->material(), this->imported());
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_mubrems() -> SPProcess
{
    return std::make_shared<MuBremsstrahlungProcess>(this->particle(),
                                                     this->imported());
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_muioni() -> SPProcess
{
    return std::make_shared<MuIonizationProcess>(
        this->particle(), this->imported(), MuIonizationProcess::Options{});
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_mupairprod() -> SPProcess
{
    return std::make_shared<MuPairProductionProcess>(
        this->particle(), this->imported(), data_.mu_production);
}

//---------------------------------------------------------------------------//
/*!
 * Warn and return a null process.
 */
auto WarnAndIgnoreProcess::operator()(argument_type) const -> result_type
{
    CELER_LOG(warning) << "Omitting " << process
                       << " from physics process list";
    return nullptr;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
