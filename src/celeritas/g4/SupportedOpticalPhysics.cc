//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SupportedOpticalPhysics.cc
//---------------------------------------------------------------------------//
#include "SupportedOpticalPhysics.hh"

#include <memory>
#include <G4Cerenkov.hh>
#include <G4EmSaturation.hh>
#include <G4LossTableManager.hh>
#include <G4OpAbsorption.hh>
#include <G4OpBoundaryProcess.hh>
#include <G4OpMieHG.hh>
#include <G4OpRayleigh.hh>
#include <G4OpWLS.hh>
#include <G4ParticleDefinition.hh>
#include <G4ProcessManager.hh>
#include <G4Scintillation.hh>
#include <G4Version.hh>

#if G4VERSION_NUMBER >= 1070
#    include <G4OpWLS2.hh>
#    include <G4OpticalParameters.hh>
#endif

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Optical physics process type.
 *
 * See Geant4's \c G4OpticalProcessIndex in G4OpticalParameters.hh for the
 * equivalent enum.
 */
enum class OpticalProcessType
{
    cherenkov,
    scintillation,
    absorption,
    rayleigh,
    mie_hg,
    boundary,
    wavelength_shifting,
    wavelength_shifting_2,
    size_,
};

/*!
 * Wrapper around a unique pointer to accommodate keeping track whether we
 * delegated ownership to Geant4. We have to assume that Geant4 won't free the
 * memory before we're done reading it...
 */
template<class T>
class ObservingUniquePtr
{
  public:
    explicit ObservingUniquePtr(std::unique_ptr<T> ptr)
        : uptr_(std::move(ptr)), ptr_{uptr_.get()}
    {
    }
    CELER_DEFAULT_MOVE_DELETE_COPY(ObservingUniquePtr);
    ~ObservingUniquePtr() = default;

    T* release() noexcept { return uptr_.release(); }
    T* release_if_owned() noexcept { return uptr_ ? uptr_.release() : ptr_; }
    operator T*() const noexcept { return ptr_; }
    T* operator->() const noexcept { return ptr_; }

  private:
    std::unique_ptr<T> uptr_;
    T* ptr_;
};

#if G4VERSION_NUMBER >= 1070
G4OpticalProcessIndex
optical_process_type_to_geant_index(OpticalProcessType value)
{
    switch (value)
    {
        case OpticalProcessType::cherenkov:
            return kCerenkov;
        case OpticalProcessType::scintillation:
            return kScintillation;
        case OpticalProcessType::absorption:
            return kAbsorption;
        case OpticalProcessType::rayleigh:
            return kRayleigh;
        case OpticalProcessType::mie_hg:
            return kMieHG;
        case OpticalProcessType::boundary:
            return kBoundary;
        case OpticalProcessType::wavelength_shifting:
            return kWLS;
        case OpticalProcessType::wavelength_shifting_2:
            return kWLS2;
        default:
            return kNoProcess;
    }
}

G4String optical_process_type_to_geant_name(OpticalProcessType value)
{
    return G4OpticalProcessName(optical_process_type_to_geant_index(value));
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Return whether a given process is active.
 *
 * Use `G4OpticalParameters` when available, otherwise use hardcoded
 * checks.
 */
bool process_is_active(
    OpticalProcessType process,
    [[maybe_unused]] SupportedOpticalPhysics::Options const& options)
{
#if G4VERSION_NUMBER >= 1070
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);
    return params->GetProcessActivation(
        optical_process_type_to_geant_name(process));
#else
    switch (process)
    {
        case OpticalProcessType::cherenkov:
            return bool(options.cherenkov);
        case OpticalProcessType::scintillation:
            return bool(options.scintillation);
        case OpticalProcessType::absorption:
            return options.absorption;
        case OpticalProcessType::rayleigh:
            return options.rayleigh_scattering;
        case OpticalProcessType::mie_hg:
            return options.mie_scattering;
        case OpticalProcessType::boundary:
            return bool(options.boundary);
        case OpticalProcessType::wavelength_shifting:
            return bool(options.wavelength_shifting);
        case OpticalProcessType::wavelength_shifting_2:
            // Technically reachable, but practically not supported pre 10.7
            CELER_ASSERT_UNREACHABLE();
        default:
            return false;
    }
#endif
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
SupportedOpticalPhysics::SupportedOpticalPhysics(Options const& options)
    : options_(options)
{
#if G4VERSION_NUMBER >= 1070
    // Use of G4OpticalParameters only from Geant4 10.7
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);

    auto activate_process = [params](G4OpticalProcessIndex i, bool flag) {
        params->SetProcessActivation(G4OpticalProcessName(i), flag);
    };

    activate_process(kCerenkov, bool(options_.cherenkov));
    activate_process(kScintillation, bool(options_.scintillation));
    activate_process(kAbsorption, options_.absorption);
    activate_process(kRayleigh, options_.rayleigh_scattering);
    activate_process(kMieHG, options_.mie_scattering);
    activate_process(kBoundary, bool(options_.boundary));
    activate_process(kWLS, bool(options_.wavelength_shifting));
    activate_process(kWLS2, bool(options_.wavelength_shifting2));

    // Cherenkov
    params->SetCerenkovStackPhotons(options_.cherenkov.stack_photons);
    params->SetCerenkovTrackSecondariesFirst(
        options_.cherenkov.track_secondaries_first);
    params->SetCerenkovMaxPhotonsPerStep(options_.cherenkov.max_photons);
    params->SetCerenkovMaxBetaChange(options_.cherenkov.max_beta_change);

    // Scintillation
    params->SetScintStackPhotons(options_.scintillation.stack_photons);
    params->SetScintTrackSecondariesFirst(
        options_.scintillation.track_secondaries_first);
    params->SetScintByParticleType(options_.scintillation.by_particle_type);
    params->SetScintFiniteRiseTime(options_.scintillation.finite_rise_time);
    params->SetScintTrackInfo(options_.scintillation.track_info);

    // WLS
    if (options_.wavelength_shifting)
    {
        params->SetWLSTimeProfile(
            to_cstring(options_.wavelength_shifting.time_profile));
    }

    // WLS2
    if (options_.wavelength_shifting2)
    {
        params->SetWLS2TimeProfile(
            to_cstring(options_.wavelength_shifting2.time_profile));
    }

    // boundary
    params->SetBoundaryInvokeSD(options_.boundary.invoke_sd);

    // Only set a global verbosity with same level for all optical processes
    params->SetVerboseLevel(options_.verbose);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Build list of available particles.
 */
void SupportedOpticalPhysics::ConstructParticle()
{
    // Eventually nothing to do here as Celeritas OpPhys won't generate
    // G4OpticalPhotons
    G4OpticalPhoton::OpticalPhotonDefinition();
}

//---------------------------------------------------------------------------//
/*!
 * Build list of available processes and models.
 */
void SupportedOpticalPhysics::ConstructProcess()
{
    auto* process_manager
        = G4OpticalPhoton::OpticalPhoton()->GetProcessManager();
    CELER_ASSERT(process_manager);

    // Add Optical Processes
    // TODO: Celeritas will eventually implement these directly (no
    // G4OpticalPhotons) so how to set up on "Celeritas-side"
    auto absorption = std::make_unique<G4OpAbsorption>();
    if (process_is_active(OpticalProcessType::absorption, options_))
    {
        process_manager->AddDiscreteProcess(absorption.release());
        CELER_LOG(debug) << "Loaded optical absorption with G4OpAbsorption "
                            "process";
    }

    auto rayleigh = std::make_unique<G4OpRayleigh>();
    if (process_is_active(OpticalProcessType::rayleigh, options_))
    {
        process_manager->AddDiscreteProcess(rayleigh.release());
        CELER_LOG(debug)
            << "Loaded optical Rayleigh scattering with G4OpRayleigh "
               "process";
    }

    auto mie = std::make_unique<G4OpMieHG>();
    if (process_is_active(OpticalProcessType::mie_hg, options_))
    {
        process_manager->AddDiscreteProcess(mie.release());
        CELER_LOG(debug) << "Loaded optical Mie (Henyey-Greenstein phase "
                            "function) scattering with G4OpMieHG "
                            "process";
    }

    // NB: boundary is also used later on in loop over particles,
    // though it's only ever applicable to G4OpticalPhotons
    auto boundary = ObservingUniquePtr{std::make_unique<G4OpBoundaryProcess>()};
#if G4VERSION_NUMBER < 1070
    boundary->SetInvokeSD(options_.boundary.invoke_sd);
#endif
    if (process_is_active(OpticalProcessType::boundary, options_))
    {
        process_manager->AddDiscreteProcess(boundary.release());
        CELER_LOG(debug)
            << "Loaded optical boundary process with G4OpBoundaryProcess "
               "process";
    }

    if (process_is_active(OpticalProcessType::wavelength_shifting, options_))
    {
        auto wls = std::make_unique<G4OpWLS>();
#if G4VERSION_NUMBER < 1070
        wls->UseTimeProfile(
            to_cstring(options_.wavelength_shifting.time_profile));
#endif
        process_manager->AddDiscreteProcess(wls.release());
        CELER_LOG(debug) << "Loaded optical wavelength shifting with G4OpWLS "
                            "process";
    }

#if G4VERSION_NUMBER >= 1070
    if (process_is_active(OpticalProcessType::wavelength_shifting_2, options_))
    {
        auto wls2 = std::make_unique<G4OpWLS2>();
        process_manager->AddDiscreteProcess(wls2.release());
        CELER_LOG(debug) << "Loaded second optical wavelength shifting with "
                            "G4OpWLS2 process ";
    }
#endif

    // Add photon-generating processes to all particles they apply to
    // TODO: Eventually replace with Celeritas step collector processes
    auto scint = ObservingUniquePtr{std::make_unique<G4Scintillation>()};
#if G4VERSION_NUMBER < 1070
    scint->SetStackPhotons(options_.scintillation.stack_photons);
    scint->SetTrackSecondariesFirst(
        options_.scintillation.track_secondaries_first);
    scint->SetScintillationByParticleType(
        options_.scintillation.by_particle_type);
    scint->SetFiniteRiseTime(options_.scintillation.finite_rise_time);
    scint->SetScintillationTrackInfo(options_.scintillation.track_info);
    // These two are not in 10.7 and newer, but defaults should be
    // sufficient for now scint->SetScintillationYieldFactor(fYieldFactor);
    // scint->SetScintillationExcitationRatio(fExcitationRatio);
#endif
    scint->AddSaturation(G4LossTableManager::Instance()->EmSaturation());

    auto cherenkov = ObservingUniquePtr{std::make_unique<G4Cerenkov>()};
#if G4VERSION_NUMBER < 1070
    cherenkov->SetStackPhotons(options_.cherenkov.stack_photons);
    cherenkov->SetTrackSecondariesFirst(
        options_.cherenkov.track_secondaries_first);
    cherenkov->SetMaxNumPhotonsPerStep(options_.cherenkov.max_photons);
    cherenkov->SetMaxBetaChangePerStep(options_.cherenkov.max_beta_change);
#endif

    auto particle_iterator = GetParticleIterator();
    particle_iterator->reset();

    while ((*particle_iterator)())
    {
        G4ParticleDefinition* p = particle_iterator->value();
        process_manager = p->GetProcessManager();
        CELER_ASSERT(process_manager);

        if (cherenkov->IsApplicable(*p)
            && process_is_active(OpticalProcessType::cherenkov, options_))
        {
            process_manager->AddProcess(cherenkov.release_if_owned());
            process_manager->SetProcessOrdering(cherenkov, idxPostStep);
            CELER_LOG(debug) << "Loaded optical Cherenkov with G4Cerenkov "
                                "process for particle "
                             << p->GetParticleName();
        }
        if (scint->IsApplicable(*p)
            && process_is_active(OpticalProcessType::scintillation, options_))
        {
            process_manager->AddProcess(scint.release_if_owned());
            process_manager->SetProcessOrderingToLast(scint, idxAtRest);
            process_manager->SetProcessOrderingToLast(scint, idxPostStep);
            CELER_LOG(debug)
                << "Loaded optical Scintillation with G4Scintillation "
                   "process for particle "
                << p->GetParticleName();
        }
        if (boundary->IsApplicable(*p)
            && process_is_active(OpticalProcessType::boundary, options_))
        {
            process_manager->SetProcessOrderingToLast(boundary, idxPostStep);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
