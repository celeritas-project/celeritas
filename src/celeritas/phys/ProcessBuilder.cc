//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ProcessBuilder.cc
//---------------------------------------------------------------------------//
#include "ProcessBuilder.hh"

#include "celeritas/em/process/BremsstrahlungProcess.hh"
#include "celeritas/em/process/ComptonProcess.hh"
#include "celeritas/em/process/EIonizationProcess.hh"
#include "celeritas/em/process/EPlusAnnihilationProcess.hh"
#include "celeritas/em/process/GammaConversionProcess.hh"
#include "celeritas/em/process/MultipleScatteringProcess.hh"
#include "celeritas/em/process/PhotoelectricProcess.hh"
#include "celeritas/em/process/RayleighProcess.hh"
#include "celeritas/io/ImportData.hh"

#include "ImportedProcessAdapter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct imported process data.
 */
ProcessBuilder::ProcessBuilder(const ImportData& data,
                               Options           options,
                               SPConstParticle   particle,
                               SPConstMaterial   material)
    : particle_(std::move(particle))
    , material_(std::move(material))
    , brem_combined_(options.brem_combined)
    , enable_lpm_(data.em_params.lpm)
    , use_integral_xs_(data.em_params.integral_approach)
{
    CELER_EXPECT(particle_);
    CELER_EXPECT(material_);

    processes_ = std::make_shared<ImportedProcesses>(data.processes);
}

//---------------------------------------------------------------------------//
//! Default destructor
ProcessBuilder::~ProcessBuilder() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct a \c Process from a given processs class.
 */
auto ProcessBuilder::operator()(ImportProcessClass ipc) -> SPProcess
{
    using IPC          = ImportProcessClass;
    using BuilderMemFn = SPProcess (ProcessBuilder::*)();

    static const std::unordered_map<IPC, BuilderMemFn> builder_funcs{
        {IPC::annihilation, &ProcessBuilder::build_annihilation},
        {IPC::compton, &ProcessBuilder::build_compton},
        {IPC::conversion, &ProcessBuilder::build_conversion},
        {IPC::e_brems, &ProcessBuilder::build_ebrems},
        {IPC::e_ioni, &ProcessBuilder::build_eioni},
        {IPC::msc, &ProcessBuilder::build_msc},
        {IPC::photoelectric, &ProcessBuilder::build_photoelectric},
        {IPC::rayleigh, &ProcessBuilder::build_rayleigh},
    };

    auto iter = builder_funcs.find(ipc);
    CELER_VALIDATE(iter != builder_funcs.end(),
                   << "cannot build unsupported EM process '"
                   << to_cstring(ipc) << "'");

    BuilderMemFn build_impl{iter->second};
    return (this->*build_impl)();
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_msc() -> SPProcess
{
    return std::make_shared<MultipleScatteringProcess>(
        particle_, material_, processes_);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_eioni() -> SPProcess
{
    EIonizationProcess::Options options;
    options.use_integral_xs = use_integral_xs_;

    return std::make_shared<EIonizationProcess>(particle_, processes_, options);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_ebrems() -> SPProcess
{
    BremsstrahlungProcess::Options options;
    options.combined_model  = brem_combined_;
    options.enable_lpm      = enable_lpm_;
    options.use_integral_xs = use_integral_xs_;

    return std::make_shared<BremsstrahlungProcess>(
        particle_, material_, processes_, options);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_photoelectric() -> SPProcess
{
    return std::make_shared<PhotoelectricProcess>(
        particle_, material_, processes_);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_compton() -> SPProcess
{
    return std::make_shared<ComptonProcess>(particle_, processes_);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_conversion() -> SPProcess
{
    GammaConversionProcess::Options options;
    options.enable_lpm = enable_lpm_;

    return std::make_shared<GammaConversionProcess>(
        particle_, processes_, options);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_rayleigh() -> SPProcess
{
    return std::make_shared<RayleighProcess>(particle_, material_, processes_);
}

//---------------------------------------------------------------------------//
auto ProcessBuilder::build_annihilation() -> SPProcess
{
    EPlusAnnihilationProcess::Options options;
    options.use_integral_xs = use_integral_xs_;

    return std::make_shared<EPlusAnnihilationProcess>(particle_, options);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
