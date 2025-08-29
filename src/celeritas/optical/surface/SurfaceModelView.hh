//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceModelView.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SurfaceModelView ...;
   \endcode
 */
class SurfaceModelView
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    inline CELER_FUNCTION SubsurfaceDirection direction() const;
    inline CELER_FUNCTION PhysSurfaceId phys_surface_id() const;
    inline CELER_FUNCTION SurfaceModelId surface_model() const;
    inline CELER_FUNCTION InternalSurfaceId internal_surface_id() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
