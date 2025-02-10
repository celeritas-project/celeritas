//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/detail/GridAccessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/grid/UniformGrid.hh"

#include "../XsGridData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for accessing grid data in different formats.
 */
class GridAccessor
{
  public:
    // Virtual destructor for polymorphic deletion
    virtual ~GridAccessor() = default;

    //! Get the x grid value at the given index
    virtual real_type x(size_type index) const = 0;

    //! Get the y grid value at the given index
    virtual real_type y(size_type index) const = 0;

    //! Get the grid size
    virtual size_type size() const = 0;

    // Calculate \f$ \Delta x_i = x_{i + 1} - x_i \f$
    inline real_type delta_x(size_type index) const;

    // Calculate \f$ \Delta y_i = y_{i + 1} - y_i \f$
    inline real_type delta_y(size_type index) const;

    // Calculate \f$ r_i - r_{i - 1} \f$
    inline real_type delta_slope(size_type index) const;
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for accessing grid data from spans.
 */
class SpanGridAccessor : public GridAccessor
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstReal = Span<real_type const>;
    //!@}

  public:
    // Construct with spans
    inline SpanGridAccessor(SpanConstReal x_values, SpanConstReal y_values);

    // Get the x grid value at the given index
    inline real_type x(size_type index) const final;

    // Get the y grid value at the given index
    inline real_type y(size_type index) const final;

    //! Get the grid size
    size_type size() const final { return x_values_.size(); }

  private:
    SpanConstReal x_values_;
    SpanConstReal y_values_;
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for accessing grid data from a cross section grid.
 */
class XsGridAccessor : public GridAccessor
{
  public:
    //!@{
    //! \name Type aliases
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct with cross section grid
    inline XsGridAccessor(XsGridData const& grid, Values const& values);

    //! Get the x grid value at the given index
    inline real_type x(size_type index) const final;

    //! Get the y grid value at the given index
    inline real_type y(size_type index) const final;

    //! Get the grid size
    size_type size() const final { return loge_grid_.size(); }

  private:
    XsGridData const& data_;
    Values const& reals_;
    UniformGrid loge_grid_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate \f$ x_{i + 1} - x_i \f$.
 */
real_type GridAccessor::delta_x(size_type index) const
{
    return this->x(index + 1) - this->x(index);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate \f$ y_{i + 1} - y_i \f$
 */
real_type GridAccessor::delta_y(size_type index) const
{
    return this->y(index + 1) - this->y(index);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate \f$ r_i - r_{i - 1} \f$.
 *
 * This calculates \f$ \Delta r_i = \frac{\Delta y_i}{\Delta x_i} -
 * \frac{\Delta y_{i - 1}}{\Delta x_{i - 1}} \f$
 */
real_type GridAccessor::delta_slope(size_type index) const
{
    CELER_EXPECT(index > 0);
    return this->delta_y(index) / this->delta_x(index)
           - this->delta_y(index - 1) / this->delta_x(index - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Contruct from spans.
 */
SpanGridAccessor::SpanGridAccessor(SpanConstReal x_values,
                                   SpanConstReal y_values)
    : x_values_(x_values), y_values_(y_values)
{
    CELER_EXPECT(x_values_.size() == y_values_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get the x grid value at the given index.
 */
real_type SpanGridAccessor::x(size_type index) const
{
    CELER_EXPECT(index < this->size());
    return x_values_[index];
}

//---------------------------------------------------------------------------//
/*!
 * Get the y grid value at the given index.
 */
real_type SpanGridAccessor::y(size_type index) const
{
    CELER_EXPECT(index < this->size());
    return y_values_[index];
}

//---------------------------------------------------------------------------//
/*!
 * Contruct from cross section grid.
 */
XsGridAccessor::XsGridAccessor(XsGridData const& grid, Values const& values)
    : data_(grid), reals_(values), loge_grid_(data_.log_energy)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the x grid value at the given index.
 */
real_type XsGridAccessor::x(size_type index) const
{
    CELER_EXPECT(index < this->size());
    return std::exp(loge_grid_[index]);
}

//---------------------------------------------------------------------------//
/*!
 * Get the y grid value at the given index.
 */
real_type XsGridAccessor::y(size_type index) const
{
    CELER_EXPECT(data_.prime_index == XsGridData::no_scaling());
    CELER_EXPECT(index < this->size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
