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
bool process_is_active(OpticalProcessType process,
                       GeantOpticalPhysicsOptions const& options)
{
    auto celer_active = [&] {
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
                return bool(options.wavelength_shifting2);
            default:
                return false;
        }
    }();
#if G4VERSION_NUMBER >= 1070
    bool g4_active = [process] {
        auto* params = G4OpticalParameters::Instance();
        CELER_ASSERT(params);
        return params->GetProcessActivation(
            optical_process_type_to_geant_name(process));
    }();
    CELER_VALIDATE(g4_active == celer_active,
                   << "inconsistent optical physics: expected "
                   << optical_process_type_to_geant_name(process) << " to be "
                   << (celer_active ? "enabled" : "disabled")
                   << " but G4OpticalParameters disagrees");
#endif
    return celer_active;
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
SupportedOpticalPhysics::SupportedOpticalPhysics(Options const& all_options)
{
    CELER_VALIDATE(all_options.optical,
                   << "cannot construct SupportedOpticalPhysics when optical "
                      "physics is disabled");
    options_ = all_options.optical.value();
    auto ensure_deactivated = [](auto& opt, char const* name, char const* why) {
        if (opt)
        {
            CELER_LOG(error) << "Ignoring incompatible " << name
                             << " optical physics: " << why;
            opt = std::nullopt;
        }
    };

    if (!(all_options.em() || all_options.muon || all_options.mucf_physics))
    {
        static char const* const why = "no EM/muon physics is enabled";
        ensure_deactivated(options_.cherenkov, "Cherenkov", why);
        ensure_deactivated(options_.scintillation, "scintillation", why);
    }

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
    if (auto& opts = options_.cherenkov)
    {
        params->SetCerenkovStackPhotons(opts->stack_photons);
        params->SetCerenkovTrackSecondariesFirst(opts->track_secondaries_first);
        params->SetCerenkovMaxPhotonsPerStep(opts->max_photons);
        params->SetCerenkovMaxBetaChange(opts->max_beta_change);
    }

    // Scintillation
    if (options_.scintillation)
    {
        params->SetScintStackPhotons(options_.scintillation->stack_photons);
        params->SetScintTrackSecondariesFirst(
            options_.scintillation->track_secondaries_first);
        params->SetScintByParticleType(
            options_.scintillation->by_particle_type);
        params->SetScintFiniteRiseTime(
            options_.scintillation->finite_rise_time);
        params->SetScintTrackInfo(options_.scintillation->track_info);
    }

    // WLS
    if (options_.wavelength_shifting)
    {
        params->SetWLSTimeProfile(
            to_cstring(options_.wavelength_shifting->time_profile));
    }

    // WLS2
    if (options_.wavelength_shifting2)
    {
        params->SetWLS2TimeProfile(
            to_cstring(options_.wavelength_shifting2->time_profile));
    }

    // boundary
    if (options_.boundary)
    {
        params->SetBoundaryInvokeSD(options_.boundary->invoke_sd);
    }

    // Only set a global verbosity with same level for all optical processes
    params->SetVerboseLevel(options_.verbose);
#else
    ensure_deactivated(
        options_.wavelength_shifting2, "WLS2", "Geant4 version is too old");
#endif

    if (options_.scintillation)
    {
        // Silence stupid Birks output
        G4LossTableManager::Instance()->EmSaturation()->SetVerbose(
            options_.verbose);
    }
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
    G4ProcessManager* process_manager
        = G4OpticalPhoton::OpticalPhoton()->GetProcessManager();
    CELER_ASSERT(process_manager);

    // Add Optical Processes
    if (process_is_active(OpticalProcessType::absorption, options_))
    {
        auto absorption = std::make_unique<G4OpAbsorption>();
        process_manager->AddDiscreteProcess(absorption.release());
        CELER_LOG(debug) << "Added optical absorption process";
    }

    if (process_is_active(OpticalProcessType::rayleigh, options_))
    {
        auto rayleigh = std::make_unique<G4OpRayleigh>();
        process_manager->AddDiscreteProcess(rayleigh.release());
        CELER_LOG(debug) << "Added optical Rayleigh scattering process";
    }

    if (process_is_active(OpticalProcessType::mie_hg, options_))
    {
        auto mie = std::make_unique<G4OpMieHG>();
        process_manager->AddDiscreteProcess(mie.release());
        CELER_LOG(debug) << "Added optical Mie (Henyey-Greenstein phase "
                            "function) scattering process";
    }

    if (process_is_active(OpticalProcessType::boundary, options_))
    {
        auto boundary
            = ObservingUniquePtr{std::make_unique<G4OpBoundaryProcess>()};
#if G4VERSION_NUMBER < 1070
        // Newer versions set these via G4OpticalParameters
        boundary->SetInvokeSD(options_.boundary->invoke_sd);
#endif
        process_manager->AddDiscreteProcess(boundary.release());
        process_manager->SetProcessOrderingToLast(boundary, idxPostStep);
        CELER_LOG(debug) << "Added optical boundary process";
    }

    if (process_is_active(OpticalProcessType::wavelength_shifting, options_))
    {
        auto wls = std::make_unique<G4OpWLS>();
#if G4VERSION_NUMBER < 1070
        wls->UseTimeProfile(
            to_cstring(options_.wavelength_shifting->time_profile));
#endif
        process_manager->AddDiscreteProcess(wls.release());
        CELER_LOG(debug) << "Added optical wavelength shifting process";
    }

#if G4VERSION_NUMBER >= 1070
    if (process_is_active(OpticalProcessType::wavelength_shifting_2, options_))
    {
        auto wls2 = std::make_unique<G4OpWLS2>();
        process_manager->AddDiscreteProcess(wls2.release());
        CELER_LOG(debug) << "Added second optical wavelength shifting process";
    }
#endif

    // Add photon-generating processes to all particles they apply to
    if (process_is_active(OpticalProcessType::scintillation, options_))
    {
        // Only update scintillation properties if there are particles that the
        // process applies to. \c G4EmSaturation requires both electron and
        // proton be defined, which is false for Celeritas optical-only runs.
        auto scint = ObservingUniquePtr{std::make_unique<G4Scintillation>()};
#if G4VERSION_NUMBER < 1070
        // Newer versions set these via G4OpticalParameters
        scint->SetStackPhotons(options_.scintillation->stack_photons);
        scint->SetTrackSecondariesFirst(
            options_.scintillation->track_secondaries_first);
        scint->SetScintillationByParticleType(
            options_.scintillation->by_particle_type);
        scint->SetFiniteRiseTime(options_.scintillation->finite_rise_time);
        scint->SetScintillationTrackInfo(options_.scintillation->track_info);
        // NOTE: scintillation yield factor and excitation ratio are not
        // present in newer versions
#endif
        scint->AddSaturation(G4LossTableManager::Instance()->EmSaturation());

        foreach_particle([&scint](G4ParticleDefinition const& p) {
            if (!scint->IsApplicable(p))
            {
                return;
            }

            G4ProcessManager* pm = p.GetProcessManager();
            CELER_ASSERT(pm);
            pm->AddProcess(scint.release_if_owned());
            pm->SetProcessOrderingToLast(scint, idxAtRest);
            pm->SetProcessOrderingToLast(scint, idxPostStep);
            CELER_LOG(debug)
                << "Added scintillation physics to " << p.GetParticleName();
        });
    }

    if (process_is_active(OpticalProcessType::cherenkov, options_))
    {
        auto cherenkov = ObservingUniquePtr{std::make_unique<G4Cerenkov>()};
#if G4VERSION_NUMBER < 1070
        // Newer versions set these via G4OpticalParameters
        cherenkov->SetStackPhotons(options_.cherenkov->stack_photons);
        cherenkov->SetTrackSecondariesFirst(
            options_.cherenkov->track_secondaries_first);
        cherenkov->SetMaxNumPhotonsPerStep(options_.cherenkov->max_photons);
        cherenkov->SetMaxBetaChangePerStep(options_.cherenkov->max_beta_change);
#endif

        foreach_particle([&cherenkov](G4ParticleDefinition const& p) {
            if (!cherenkov->IsApplicable(p))
            {
                return;
            }

            G4ProcessManager* pm = p.GetProcessManager();
            CELER_ASSERT(pm);
            pm->AddProcess(cherenkov.release_if_owned());
            pm->SetProcessOrdering(cherenkov, idxPostStep);
            CELER_LOG(debug)
                << "Added Cherenkov physics to " << p.GetParticleName();
        });
    }
}

//---------------------------------------------------------------------------//
//! Apply a function to every particle
template<class F>
void SupportedOpticalPhysics::foreach_particle(F&& apply) const
{
    auto particle_iterator = this->GetParticleIterator();
    particle_iterator->reset();

    while ((*particle_iterator)())
    {
        G4ParticleDefinition* p = particle_iterator->value();
        CELER_ASSERT(p);
        apply(*p);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
