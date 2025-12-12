//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Toroid.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Z-aligned Elliptical Toroid.
 *
 * An elliptical toroid is a shape created by revolving an axis-aligned ellipse
 * around a central axis. This shape can be used in everything from pipe bends
 * to tokamaks in fusion reactors. Possesses a major radius r, and ellipse
 * radii a and b, as shown in the below diagram:
 *     ___   _________   ___
 *   /  |  \           /     \
 *  /   b   \         /       \
 * |    |    |       |         |
 * |-a--+    |   o-----r--+    |
 *  \       /         \       /
 *   \     /           \     /
 *     ⁻⁻⁻   ⁻⁻⁻⁻⁻⁻⁻⁻⁻   ⁻⁻⁻
 *
 * This torus can be defined with the following quartic equation:
 * \f[
 *   (x^2 + y^2 + p*y^2 + B_0) - A_0 * (x^2 + y^2) = 0
 * \f]
 * where \f[p = a^2/b^2, A_0 = 4*r^2, and B_0 = (r^2-a^2)\f].
 */
}  // namespace celeritas
