//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/MuBremsDiffXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/ElementView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the differential cross section for muon bremsstrahlung.
 *
 * The differential cross section can be written as
 * \f[
   \difd{\sigma}{\epsilon} = \frac{16}{3} \alpha N_A (\frac{m}{\mu}
   r_e)^2 \frac{1}{\epsilon A} Z(Z \Phi_n + \Phi_e) (1 - v + \frac{3}{4} v^2),
 * \f]
 * where \f$ \epsilon \f$ is the photon energy, \f$ \alpha \f$ is the fine
 * structure constant, \f$ N_A \f$ is Avogadro's number, \f$ m \f$ is the
 * electron mass, \f$ \mu \f$ is the muon mass, \f$ r_e \f$ is the classical
 * electron radius, \f$ Z \f$ is the atomic number, and \f$ A \f$ is the atomic
 * mass. \f$ v = \epsilon / E \f$ is the relative energy transfer, where \f$ E
 * \f$ is the total energy of the incident muon.
 *
 * The contribution to the cross section from the nucleus is given by
 * \f[
   \Phi_n = \ln \frac{B Z^{-1/3} (\mu + \delta(D'_n \sqrt{e} - 2))}{D'_n (m +
   \delta \sqrt{e} B Z^{-1/3})} \ ,
 * \f]
 * where \f$ \delta = \frac{\mu^2 v}{2(E - \epsilon)}\f$ is the minimum
 * momentum transfer and \f$ D'_n \f$ is the correction to the nuclear form
 * factor.
 *
 * The contribution to the cross section from electrons is given by
 * \f[
   \Phi_e = \ln \frac{B' Z^{-2/3} \mu}{\left(1 + \frac{\delta \mu}{m^2
   \sqrt{e}}\right)(m + \delta \sqrt{e} B' Z^{-2/3})} \ .
 * \f]
 *
 * The constants \f$ B \f$ and \f$ B' \f$ were calculated using the
 * Thomas-Fermi model. In the case of hydrogen, where the Thomas-Fermi model
 * does not serve as a good approximation, the exact values of the constants
 * were calculated analytically.
 *
 * This performs the same calculation as in Geant4's \c
 * G4MuBremsstrahlungModel::ComputeDMicroscopicCrossSection() and as described
 * in section 11.2.1 of \cite{g4prm}. The formulae are taken
 * mainly from \citet{kelner-muxs-1995, https://cds.cern.ch/record/288828/}.
 */
class MuBremsDiffXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    // Construct with incident particle data and current element
    inline CELER_FUNCTION MuBremsDiffXsCalculator(ElementView const& element,
                                                  Energy inc_energy,
                                                  Mass inc_mass,
                                                  Mass electron_mass);

    // Compute cross section of exiting gamma energy
    inline CELER_FUNCTION real_type operator()(Energy energy);

  private:
    //// DATA ////

    // Atomic number of the current element
    int atomic_number_;
    // Atomic mass of the current element
    real_type atomic_mass_;
    // \f$ Z^{-1/3} \f$
    real_type inv_cbrt_z_;
    // Energy of the incident particle
    real_type inc_energy_;
    // Mass of the incident particle
    real_type inc_mass_;
    // Square of the incident particle mass
    real_type inc_mass_sq_;
    // Total energy of the incident particle
    real_type total_energy_;
    // Mass of an electron
    real_type electron_mass_;
    // Correction to the nuclear form factor
    real_type d_n_;
    // Constant in the radiation logarithm
    real_type b_;
    // Constant in the inelastic radiation logarithm
    real_type b_prime_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with incident particle data and current element.
 */
CELER_FUNCTION
MuBremsDiffXsCalculator::MuBremsDiffXsCalculator(ElementView const& element,
                                                 Energy inc_energy,
                                                 Mass inc_mass,
                                                 Mass electron_mass)
    : atomic_number_(element.atomic_number().unchecked_get())
    , atomic_mass_(value_as<units::AmuMass>(element.atomic_mass()))
    , inv_cbrt_z_(1 / element.cbrt_z())
    , inc_energy_(value_as<Energy>(inc_energy))
    , inc_mass_(value_as<Mass>(inc_mass))
    , inc_mass_sq_(ipow<2>(inc_mass_))
    , total_energy_(inc_energy_ + inc_mass_)
    , electron_mass_(value_as<Mass>(electron_mass))
{
    CELER_EXPECT(inc_energy_ > 0);

    d_n_ = real_type(1.54) * std::pow(atomic_mass_, real_type(0.27));
    if (atomic_number_ == 1)
    {
        // Constants calculated calculated analytically
        b_ = real_type(202.4);
        b_prime_ = 446;
    }
    else
    {
        // Constants calculated using the Thomas-Fermi model
        b_ = 183;
        b_prime_ = 1429;
        d_n_ = std::pow(d_n_, 1 - real_type(1) / atomic_number_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Compute the differential cross section per atom at the given photon energy.
 */
CELER_FUNCTION
real_type MuBremsDiffXsCalculator::operator()(Energy energy)
{
    CELER_EXPECT(energy > zero_quantity());

    using namespace celeritas::constants;

    if (value_as<Energy>(energy) >= inc_energy_)
    {
        return 0;
    }

    // Calculate the relative energy transfer
    real_type v = value_as<Energy>(energy) / total_energy_;

    // Calculate the minimum momentum transfer
    real_type delta = real_type(0.5) * inc_mass_sq_ * v
                      / (total_energy_ - value_as<Energy>(energy));

    // Calculate the contribution to the cross section from the nucleus
    real_type phi_n = clamp_to_nonneg(std::log(
        b_ * inv_cbrt_z_ * (inc_mass_ + delta * (d_n_ * sqrt_euler - 2))
        / (d_n_ * (electron_mass_ + delta * sqrt_euler * b_ * inv_cbrt_z_))));

    // Photon energy above which there is no contribution from electrons
    real_type energy_max_prime = total_energy_
                                 / (1
                                    + real_type(0.5) * inc_mass_sq_
                                          / (electron_mass_ * total_energy_));

    // Calculate the contribution to the cross section from electrons
    real_type phi_e = 0;
    if (value_as<Energy>(energy) < energy_max_prime)
    {
        real_type inv_cbrt_z_sq = ipow<2>(inv_cbrt_z_);
        phi_e = clamp_to_nonneg(std::log(
            b_prime_ * inv_cbrt_z_sq * inc_mass_
            / ((1 + delta * inc_mass_ / (ipow<2>(electron_mass_) * sqrt_euler))
               * (electron_mass_
                  + delta * sqrt_euler * b_prime_ * inv_cbrt_z_sq))));
    }

    // Calculate the differential cross section
    return 16 * alpha_fine_structure * na_avogadro
           * ipow<2>(electron_mass_ * r_electron) * atomic_number_
           * (atomic_number_ * phi_n + phi_e)
           * (1 - v * (1 - real_type(0.75) * v))
           / (3 * inc_mass_sq_ * value_as<Energy>(energy) * atomic_mass_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
