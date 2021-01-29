//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VectorUtils.hh
//! \brief Helper functions for std::vector
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Move the given span to the end of another vector.
template<class T, std::size_t N, class U>
inline Span<U> extend(Span<T, N> ext, std::vector<U>* base);

//---------------------------------------------------------------------------//
// Move the given extension to the end of another vector.
template<class T>
inline Span<T> extend(const std::vector<T>& ext, std::vector<T>* base);

//---------------------------------------------------------------------------//
// Move the given extension to the end of another vector.
template<class T>
inline Span<T> move_extend(std::vector<T>&& ext, std::vector<T>* base);

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "VectorUtils.i.hh"
