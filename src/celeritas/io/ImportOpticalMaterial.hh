//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalMaterial.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <vector>

#include "corecel/inp/Grid.hh"

#include "ImportUnits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store optical material properties for Rayleigh scattering.
 *
 * The isothermal compressibility is used to calculate the Rayleigh mean free
 * path if no mean free paths are provided.
 */
struct ImportOpticalRayleigh
{
    double scale_factor{1};  //!< Scale the scattering length (optional)
    double compressibility{0};  //!< Isothermal compressibility
                                //!< [len-time^2/mass]

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return scale_factor > 0 && compressibility > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store common optical material properties.
 */
struct ImportOpticalProperty
{
    inp::Grid refractive_index;

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return static_cast<bool>(refractive_index);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical photon wavelength shifting properties.
 *
 * The component vector represents the relative population as a function of the
 * re-emission energy. It is used to define an inverse CDF needed to sample the
 * re-emitted optical photon energy.
 */
struct ImportWavelengthShift
{
    double mean_num_photons{};  //!< Mean number of re-emitted photons
    double time_constant{};  //!< Time delay between absorption and re-emission
    inp::Grid component;  //!< Re-emission population [MeV, unitless]

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return mean_num_photons > 0 && time_constant > 0 && component;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store Mie scattering properties (Henyey–Greenstein model).
 */
struct ImportMie
{
    //! Henyey–Greenstein "g" parameter for forward scattering
    double forward_g{};
    //! Henyey–Greenstein "g" parameter for backward scattering
    double backward_g{};
    //! Fraction of forward vs backward scattering
    double forward_ratio{};

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return forward_ratio >= 0 && forward_ratio <= 1 && forward_g >= -1
               && forward_g <= 1 && backward_g >= -1 && backward_g <= 1;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties.
 *
 * \todo boolean for enabling cherenkov in the material?? DUNE e.g. disables
 * cherenkov globally.
 */
struct ImportOpticalMaterial
{
    ImportOpticalProperty properties;

    //! Whether minimal useful data is stored
    explicit operator bool() const { return static_cast<bool>(properties); }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
