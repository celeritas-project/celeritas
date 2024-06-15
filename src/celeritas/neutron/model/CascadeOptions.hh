//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/CascadeOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
//#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Configuration options for the Bertini cascade model.
 */
struct CascadeOptions
{
    /// Top-level Configuration flags

    //! The flag for using the PreCompound model
    bool use_precompound = false;

    //! The flag for using ABLA data
    bool use_abla = false;

    //! The flag for using the three body momentum
    bool use_three_body_mom = false;

    //! The flag for using the phase space
    bool use_phase_space = false;

    //! The flag for applying coalescence
    bool do_coalescence = true;

    //! The probability for the pion-nucleon absorption
    real_type prob_pion_absorption = 0;

    /// Nuclear structure parameters

    //! The flag for using two parameters for the nuclear radius
    bool use_two_params = false;

    //! The scale for the nuclear radius in [femtometer]
    real_type radius_scale = 2.81967;

    //! The radius scale for small A (A < 4)
    real_type radius_small = 8;

    //! The radius scale for alpha (A = 4)
    real_type radius_alpha = 0.7;

    //! The trailing factor for the nuclear radius
    real_type radius_trailing = 0;

    //! The scale for the Fermi radius in [MeV/c]
    real_type fermi_scale = 1932;

    //! The scale for cross sections
    real_type xsec_scale = 1;

    //! The scale for the quasi-deuteron cross section of gamma
    real_type gamma_qd_scale = 1;

    /// Final-state clustering cuts

    //! The final state clustering cut (2 clusters)
    real_type dp_max_doublet = 0.09;

    //! The final state clustering cut (3 clusters)
    real_type dp_max_triplet = 0.108;

    //! The final state clustering cut (4 clusters)
    real_type dp_max_alpha = 0.115;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
      return  (prob_pion_absorption >= 0)
	       && (radius_scale > 0)
	       && (radius_small > 0)
	       && (radius_alpha > 0)
	       && (radius_trailing >= 0)
	       && (fermi_scale > 0)
	       && (xsec_scale > 0)
	       && (gamma_qd_scale > 0)
	       && (dp_max_doublet > 0)
	       && (dp_max_triplet > 0)
	       && (dp_max_alpha > 0);
        // clang-format on
    }
};

//---------------------------------------------------------------------------//
//! Equality operator
constexpr bool operator==(CascadeOptions const& a, CascadeOptions const& b)
{
    // clang-format off
    return a.use_precompound == b.use_precompound
           && a.use_abla == b.use_abla
           && a.use_three_body_mom == b.use_three_body_mom
           && a.use_phase_space == b.use_phase_space
           && a.do_coalescence == b.do_coalescence
           && a.prob_pion_absorption == b.prob_pion_absorption
           && a.use_two_params == b.use_two_params
           && a.radius_scale == b.radius_scale
           && a.radius_small == b.radius_small
           && a.radius_alpha == b.radius_alpha
           && a.radius_trailing == b.radius_trailing
           && a.fermi_scale == b.fermi_scale
           && a.xsec_scale == b.xsec_scale
           && a.gamma_qd_scale == b.gamma_qd_scale
           && a.dp_max_doublet == b.dp_max_doublet
           && a.dp_max_triplet == b.dp_max_triplet
           && a.dp_max_alpha == b.dp_max_alpha;
    // clang-format on
}

//---------------------------------------------------------------------------//
// Throw a runtime assertion if any of the input is invalid
void validate_input(CascadeOptions const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
