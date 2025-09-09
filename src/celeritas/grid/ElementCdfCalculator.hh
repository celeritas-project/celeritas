//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ElementCdfCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "celeritas/inp/Grid.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the microscopic cross section CDF table for a material.
 *
 * The CDF is used to sample an element from a material for a discrete
 * interaction. It's computed for each energy bin across all constituent
 * elements.
 */
class ElementCdfCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstElement = Span<MatElementComponent const>;
    using XsTable = Model::XsTable;
    //!@}

  public:
    // Construct from elements in a material
    explicit ElementCdfCalculator(SpanConstElement elements);

    // Calculate the CDF in place from the microscopic cross sections
    void operator()(XsTable& grids);

  private:
    SpanConstElement elements_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
