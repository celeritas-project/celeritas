//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ResponseTallyParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/grid/UniformGrid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    ResponseTallyParams ...;
   \endcode
 */
struct ResponseTallyParams
{
    //!@{
    //! \name Type aliases
    UniformGridData x_grid;
    UniformGridData y_grid;
    UniformGridData z_grid;
    UniformGridData time_grid;
    size_t num_detectors;
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
