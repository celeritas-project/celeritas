//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/EnergyLossHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Type of energy loss distribution sampling to perform
enum class EnergyLossFluctuationModel
{
    none,  //!< Low density material: do not adjust
    gamma,  //!< Heavy particles with small mean energy loss
    gaussian,  //!< Heavy particles
    urban,  //!< Thin layers
};

//---------------------------------------------------------------------------//
/*!
 * Precalculate energy loss fluctuation properties.
 *
 * Fluctuations in the energy loss of charged particles over a given thickness
 * of material arise from statistical variation in both the number of
 * collisions and the energy lost in each collision. Above a given energy
 * threshold, fluctuations are simulated through the explicit sampling of
 * secondaries. However, the continuous energy loss below the cutoff energy
 * also has fluctuations, and these are not taken into account in the
 * calculation of the mean loss. For continuous energy loss, fluctuation models
 * are used to sample the actual restricted energy loss given the mean loss.
 *
 * Different models are used depending on the value of the parameter
 * \f$ \kappa = \xi / T_{max} \f$, the ratio of the mean energy loss to the
 * maximum allowed energy transfer in a single collision.
 *
 * For large \f$ \kappa \f$,
 * when the particle loses all or most of its energy along the step, the number
 * of collisions is large and the straggling function can be approximated by a
 * Gaussian distribution.
 *
 * Otherwise, the Urban model for energy loss fluctuations in thin layers is
 * used.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4UniversalFluctuation, as documented in section 7.3 in the Geant4 Physics
 * Reference Manual \cite{g4prm} and in PHYS332 and PHYS333 in the GEANT3
 * manual \citep{geant3-1993,
 * https://geant4-userdoc.web.cern.ch/UsersGuides/PhysicsReferenceManual/html/index.html}.
 */
class EnergyLossHelper
{
  public:
    //!@{
    //! \name Type aliases
    using FluctuationRef = NativeCRef<FluctuationData>;
    using Energy = units::MevEnergy;
    using EnergySq = RealQuantity<UnitProduct<units::Mev, units::Mev>>;
    using Mass = units::MevMass;
    using Charge = units::ElementaryCharge;
    using Model = EnergyLossFluctuationModel;
    using Real2 = Array<real_type, 2>;
    //!@}

  public:
    // Construct from model parameters, incident particle, and mean energy loss
    inline CELER_FUNCTION EnergyLossHelper(FluctuationRef const& shared,
                                           CutoffView const& cutoffs,
                                           MaterialTrackView const& material,
                                           ParticleTrackView const& particle,
                                           Energy mean_loss,
                                           real_type step_length);

    //// STATIC ACCESSORS ////

    //! Atomic energy level corresponding to outer electrons (E_0)
    static CELER_CONSTEXPR_FUNCTION Energy ionization_energy()
    {
        return units::MevEnergy{1e-5};
    }

    //// ACCESSORS ////

    //! Shared data
    CELER_FORCEINLINE_FUNCTION FluctuationRef const& shared() const
    {
        return shared_;
    }

    //! Current material
    CELER_FORCEINLINE_FUNCTION MaterialTrackView const& material() const
    {
        return material_;
    }

    //! Type of model to select
    CELER_FORCEINLINE_FUNCTION Model model() const { return model_; }

    //! Input mean loss
    CELER_FUNCTION Energy mean_loss() const { return Energy{mean_loss_}; }

    //! Maximum allowable energy loss
    CELER_FUNCTION Energy max_energy() const
    {
        CELER_EXPECT(model_ != Model::none);
        return Energy{max_energy_};
    }

    //! Kinematics value: square of fractional speed of light
    CELER_FUNCTION real_type beta_sq() const
    {
        CELER_ENSURE(beta_sq_ > 0);
        return beta_sq_;
    }

    //! Bohr variance
    CELER_FUNCTION EnergySq bohr_variance() const
    {
        CELER_EXPECT(model_ != Model::none);
        CELER_ENSURE(bohr_var_ > 0);
        return EnergySq{bohr_var_};
    }

    //! Kinematics value for electrons: 2 * mass * beta^2 * gamma^2
    CELER_FUNCTION Mass two_mebsgs() const
    {
        CELER_ENSURE(two_mebsgs_ > 0);
        return Mass{two_mebsgs_};
    }

  private:
    //// DATA ////

    // Shared properties of the fluctuation model
    FluctuationRef const& shared_;
    // Current material
    MaterialTrackView const& material_;
    // Model to use for the given step
    Model model_{Model::none};
    // Average energy loss calculated from the tables
    real_type mean_loss_{0};

    // Square of the ratio of the particle velocity to the speed of light
    real_type beta_sq_{0};
    // Smaller of the delta ray production cut and maximum energy transfer
    real_type max_energy_{0};
    // two_mebg = 2 * m_e c^2 * beta^2 * gamma^2
    real_type two_mebsgs_{0};

    // Bohr's variance
    real_type bohr_var_{0};

    //// CONSTANTS ////

    //! Lower limit on the number of interactions in a step (kappa)
    static CELER_CONSTEXPR_FUNCTION size_type min_kappa() { return 10; }

    //! Minimum mean energy loss required to sample fluctuations
    static CELER_CONSTEXPR_FUNCTION real_type min_energy()
    {
        return value_as<Energy>(EnergyLossHelper::ionization_energy());
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from model parameters, incident particle, and mean energy loss.
 */
CELER_FUNCTION
EnergyLossHelper::EnergyLossHelper(FluctuationRef const& shared,
                                   CutoffView const& cutoffs,
                                   MaterialTrackView const& material,
                                   ParticleTrackView const& particle,
                                   Energy mean_loss,
                                   real_type step_length)
    : shared_(shared)
    , material_(material)
    , mean_loss_(value_as<Energy>(mean_loss))
{
    CELER_EXPECT(shared_);
    CELER_EXPECT(mean_loss_ > 0);
    CELER_EXPECT(step_length > 0);

    if (mean_loss_ < this->min_energy())
    {
        model_ = Model::none;
        return;
    }

    constexpr real_type half = 0.5;
    real_type const gamma = particle.lorentz_factor();
    beta_sq_ = particle.beta_sq();
    two_mebsgs_ = 2 * value_as<Mass>(shared_.electron_mass) * beta_sq_
                  * ipow<2>(gamma);

    // Maximum possible energy transfer to an electron in a single collision
    real_type max_energy_transfer;
    real_type mass_ratio = 1;
    if (particle.particle_id() == shared_.electron_id)
    {
        max_energy_transfer = half * value_as<Energy>(particle.energy());
    }
    else
    {
        mass_ratio = value_as<Mass>(shared_.electron_mass)
                     / value_as<Mass>(particle.mass());
        max_energy_transfer = two_mebsgs_
                              / (1 + mass_ratio * (2 * gamma + mass_ratio));
    }
    max_energy_ = min(value_as<Energy>(cutoffs.energy(shared_.electron_id)),
                      max_energy_transfer);

    if (max_energy_ <= value_as<Energy>(this->ionization_energy()))
    {
        model_ = Model::none;
        return;
    }

    // Units: [len^2][MeV c^2][1/len^3][e-][MeV][len] = MeV^2
    // assuming implicit 1/c^2 in the formula
    bohr_var_ = 2 * constants::pi * ipow<2>(constants::r_electron)
                * value_as<Mass>(shared_.electron_mass)
                * material_.material_record().electron_density()
                * ipow<2>(value_as<Charge>(particle.charge())) * max_energy_
                * step_length * (1 / beta_sq_ - half);

    if (mass_ratio >= 1 || mean_loss_ < this->min_kappa() * max_energy_
        || max_energy_transfer > 2 * max_energy_)
    {
        model_ = Model::urban;
    }
    else if (ipow<2>(mean_loss_) >= 4 * bohr_var_)
    {
        model_ = Model::gaussian;
    }
    else
    {
        model_ = Model::gamma;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
