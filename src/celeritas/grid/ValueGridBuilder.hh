//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/Grid.hh"

namespace celeritas
{
class XsGridInserter;
struct XsGridRecord;

//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing on-device physics data for a single material.
 *
 * These builder classes are presumed to have a short/temporary lifespan and
 * should not be retained after the setup phase.
 */
class ValueGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using ValueGridId = ItemId<struct XsGridRecord>;
    using VecDbl = inp::Grid::VecDbl;
    //!@}

    struct GridInput
    {
        double emin{0};
        double emax{0};
        VecDbl value;
    };

  public:
    //! Virtual destructor for polymorphic deletion
    virtual ~ValueGridBuilder() = 0;

    //! Construct the grid given a mutable reference to a store
    virtual ValueGridId build(XsGridInserter) const = 0;

  protected:
    ValueGridBuilder() = default;

    //!@{
    //! Prevent copy/move except by daughters that know what they're doing
    ValueGridBuilder(ValueGridBuilder const&) = default;
    ValueGridBuilder& operator=(ValueGridBuilder const&) = default;
    ValueGridBuilder(ValueGridBuilder&&) = default;
    ValueGridBuilder& operator=(ValueGridBuilder&&) = default;
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Build a physics array for EM process cross sections.
 *
 * This array has a uniform grid in log(E) and a special value at or above
 * which the input cross sections are scaled by E.
 */
class ValueGridXsBuilder final : public ValueGridBuilder
{
  public:
    // Construct from imported data
    static std::unique_ptr<ValueGridXsBuilder>
    from_geant(inp::Grid const& lower, inp::Grid const& upper);

    // Construct from just scaled cross sections
    static std::unique_ptr<ValueGridXsBuilder>
    from_scaled(inp::Grid const& upper);

    // Construct
    ValueGridXsBuilder(GridInput lower, GridInput upper);

    // Construct in the given store
    ValueGridId build(XsGridInserter) const final;

  private:
    GridInput lower_;
    GridInput upper_;
};

//---------------------------------------------------------------------------//
/*!
 * Build a physics vector for energy loss and other quantities.
 *
 * This vector is still uniform in log(E).
 *
 * \todo Currently this builds and inserts an \c XsGridRecord, but should build
 * a \c UniformGridRecord since there are no scaled values.
 */
class ValueGridLogBuilder : public ValueGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using UPLogBuilder = std::unique_ptr<ValueGridLogBuilder>;
    //!@}

  public:
    // Construct from full grids
    static UPLogBuilder from_geant(VecDbl const& x, VecDbl const& y);

    // Construct from full grids
    static UPLogBuilder from_geant(inp::Grid const&);

    // Construct
    ValueGridLogBuilder(GridInput);

    // Construct in the given store
    ValueGridId build(XsGridInserter) const final;

  private:
    GridInput grid_;
};

//---------------------------------------------------------------------------//
/*!
 * Special cases for indicating *only* on-the-fly cross sections.
 *
 * Currently this should be thrown just for processes and models specified in
 * \c HardwiredModels as needed for EPlusAnnihilationProcess, which has *only*
 * on-the-fly cross section calculation.
 *
 * This class is needed so that the process has at least one "builder"; but it
 * always returns an invalid ValueGridId.
 */
class ValueGridOTFBuilder final : public ValueGridBuilder
{
  public:
    // Don't construct anything
    ValueGridId build(XsGridInserter) const final;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
