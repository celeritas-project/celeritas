//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportParticle.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store particle data.
 */
struct ImportParticle
{
    //!@{
    //! \name Type aliases
    using PdgInt = int;
    //!@}

    std::string name;
    PdgInt pdg{0};
    double mass{0};  //!< [MeV]
    double charge{0};  //!< [Multiple of electron charge value]
    double spin{0};  //!< [Multiple of hbar]
    double lifetime{0};  //!< [time]
    bool is_stable{false};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
