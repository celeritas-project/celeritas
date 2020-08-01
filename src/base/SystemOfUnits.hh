//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file base/SystemOfUnits.hh
//---------------------------------------------------------------------------//
/**
 * @brief   System of units.
 * @file    Geant/core/SystemOfUnits.hpp
 * @author  M Novak, A Ribon
 * @date    december 2015
 *
 * Original file was taken from CLHEP and changed according to GeantV.
 * Authors of the original version: M. Maire, S. Giani
 *
 * The basic units are :
 * \li  centimeter              (centimeter)
 * \li  second                  (second)
 * \li  giga electron volt      (GeV)
 * \li  positron charge         (eplus)
 * \li  degree Kelvin           (kelvin)
 * \li  the amount of substance (mole)
 * \li  luminous intensity      (candela)
 * \li  radian                  (radian)
 * \li  steradian               (steradian)
 *
 * It's a non exhaustive list of derived and pratical units (i.e. mostly the SI
 * units).
 *
 * The value of \f$ \pi \f$ (kPi) is defined here as it is needed for radian to degree
 * conversion. The other physical constants are defined in the header file :
 * Geant/core/PhysicalConstants.hpp
 *
 * You can add your own units.
 *
 */
#pragma once

namespace celeritas {
namespace units {
//
// (kPi is used here (degree) so we declare here and not among the constants.)
//
static constexpr double kPi       = 3.14159265358979323846;
static constexpr double kTwoPi    = 2. * kPi;
static constexpr double kHalfPi   = kPi / 2.;
static constexpr double kPiSquare = kPi * kPi;

//
// Length [L]
//
static constexpr double centimeter  = 1.;
static constexpr double centimeter2 = centimeter * centimeter;
static constexpr double centimeter3 = centimeter * centimeter * centimeter;

static constexpr double millimeter  = 0.1 * centimeter;
static constexpr double millimeter2 = millimeter * millimeter;
static constexpr double millimeter3 = millimeter * millimeter * millimeter;

static constexpr double meter  = 100. * centimeter;
static constexpr double meter2 = meter * meter;
static constexpr double meter3 = meter * meter * meter;

static constexpr double kilometer  = 1000. * meter;
static constexpr double kilometer2 = kilometer * kilometer;
static constexpr double kilometer3 = kilometer * kilometer * kilometer;

static constexpr double parsec = 3.0856775807e+16 * meter;

static constexpr double micrometer = 1.e-6 * meter;
static constexpr double nanometer  = 1.e-9 * meter;
static constexpr double angstrom   = 1.e-10 * meter;
static constexpr double fermi      = 1.e-15 * meter;

static constexpr double barn      = 1.e-28 * meter2;
static constexpr double millibarn = 1.e-3 * barn;
static constexpr double microbarn = 1.e-6 * barn;
static constexpr double nanobarn  = 1.e-9 * barn;
static constexpr double picobarn  = 1.e-12 * barn;

// symbols
static constexpr double nm = nanometer;
static constexpr double um = micrometer;

static constexpr double mm  = millimeter;
static constexpr double mm2 = millimeter2;
static constexpr double mm3 = millimeter3;

static constexpr double cm  = centimeter;
static constexpr double cm2 = centimeter2;
static constexpr double cm3 = centimeter3;

static constexpr double liter = 1.e+3 * cm3;
static constexpr double L     = liter;
static constexpr double dL    = 1.e-1 * liter;
static constexpr double cL    = 1.e-2 * liter;
static constexpr double mL    = 1.e-3 * liter;

static constexpr double m  = meter;
static constexpr double m2 = meter2;
static constexpr double m3 = meter3;

static constexpr double km  = kilometer;
static constexpr double km2 = kilometer2;
static constexpr double km3 = kilometer3;

static constexpr double pc = parsec;

//
// Angle
//
static constexpr double radian      = 1.;
static constexpr double milliradian = 1.e-3 * radian;
static constexpr double degree      = (kPi / 180.0) * radian;

static constexpr double steradian = 1.;

// symbols
static constexpr double rad  = radian;
static constexpr double mrad = milliradian;
static constexpr double sr   = steradian;
static constexpr double deg  = degree;

//
// Time [T]
//
static constexpr double second      = 1.;
static constexpr double millisecond = 1.e-3 * second;
static constexpr double microsecond = 1.e-6 * second;
static constexpr double nanosecond  = 1.e-9 * second;
static constexpr double picosecond  = 1.e-12 * second;

static constexpr double hertz     = 1. / second;
static constexpr double kilohertz = 1.e+3 * hertz;
static constexpr double megahertz = 1.e+6 * hertz;

// symbols
static constexpr double s  = second;
static constexpr double ms = millisecond;
static constexpr double ns = nanosecond;

//
// Electric charge [Q]
//
static constexpr double eplus   = 1.;              // positron charge
static constexpr double e_SI    = 1.602176487e-19; // positron charge in coulomb
static constexpr double coulomb = eplus / e_SI;    // coulomb = 6.24150 e+18 * eplus

//
// Energy [E]
//
static constexpr double gigaelectronvolt = 1.;
static constexpr double megaelectronvolt = 1.e-3 * gigaelectronvolt;
static constexpr double kiloelectronvolt = 1.e-6 * gigaelectronvolt;
static constexpr double electronvolt     = 1.e-9 * gigaelectronvolt;
static constexpr double teraelectronvolt = 1.e+3 * gigaelectronvolt;
static constexpr double petaelectronvolt = 1.e+6 * gigaelectronvolt;

static constexpr double joule = electronvolt / e_SI; // joule = 6.24150 e+9 * GeV

// symbols
static constexpr double eV  = electronvolt;
static constexpr double keV = kiloelectronvolt;
static constexpr double MeV = megaelectronvolt;
static constexpr double GeV = gigaelectronvolt;
static constexpr double TeV = teraelectronvolt;
static constexpr double PeV = petaelectronvolt;

//
// Mass [E][T^2][L^-2]
//
static constexpr double kilogram  = joule * second * second / (meter * meter);
static constexpr double gram      = 1.e-3 * kilogram;
static constexpr double milligram = 1.e-3 * gram;

// symbols
static constexpr double kg = kilogram;
static constexpr double g  = gram;
static constexpr double mg = milligram;

//
// Power [E][T^-1]
//
static constexpr double watt = joule / second; // watt = 6.24150 e+9 * GeV/s

//
// Force [E][L^-1]
//
static constexpr double newton = joule / meter; // newton = 6.24150 e+7 * GeV/cm

//
// Pressure [E][L^-3]
//
static constexpr double pascal     = newton / m2;     // pascal = 6.24150 e+3 * GeV/cm3
static constexpr double bar        = 1.e+5 * pascal;  // bar    = 6.24150 e+8 * GeV/cm3
static constexpr double atmosphere = 101325 * pascal; // atm    = 6.32420 e+8 * GeV/cm3

//
// Electric current [Q][T^-1]
//
static constexpr double ampere      = coulomb / second; // ampere = 6.24150 e+18 * eplus/s
static constexpr double milliampere = 1.e-3 * ampere;
static constexpr double microampere = 1.e-6 * ampere;
static constexpr double nanoampere  = 1.e-9 * ampere;

//
// Electric potential [E][Q^-1]
//
static constexpr double gigavolt = gigaelectronvolt / eplus;
static constexpr double megavolt = 1.e-3 * gigavolt;
static constexpr double kilovolt = 1.e-3 * megavolt;
static constexpr double volt     = 1.e-3 * kilovolt;

//
// Electric resistance [E][T][Q^-2]
//
static constexpr double ohm = volt / ampere; // ohm = 1.60217e-18*(GeV/eplus)/(eplus/s)

//
// Electric capacitance [Q^2][E^-1]
//
static constexpr double farad = coulomb / volt; // farad = 6.24150e+27 * eplus/gigavolt
static constexpr double millifarad = 1.e-3 * farad;
static constexpr double microfarad = 1.e-6 * farad;
static constexpr double nanofarad  = 1.e-9 * farad;
static constexpr double picofarad  = 1.e-12 * farad;

//
// Magnetic Flux [T][E][Q^-1]
//
static constexpr double weber = volt * second; // weber = 1.e-9*gigavolt*s

//
// Magnetic Field [T][E][Q^-1][L^-2]
//
static constexpr double tesla = volt * second / meter2; // tesla =1.e-13*gigavolt*s/cm2

static constexpr double gauss     = 1.e-4 * tesla;
static constexpr double kilogauss = 1.e-1 * tesla;

//
// Inductance [T^2][E][Q^-2]
//
static constexpr double henry = weber / ampere; // henry = 1.60217e-28*GeV*(s/eplus)^2

//
// Temperature
//
static constexpr double kelvin = 1.;

//
// Amount of substance
//
static constexpr double mole = 1.;

//
// Activity [T^-1]
//
static constexpr double becquerel     = 1. / second;
static constexpr double curie         = 3.7e+10 * becquerel;
static constexpr double kilobecquerel = 1.e+3 * becquerel;
static constexpr double megabecquerel = 1.e+6 * becquerel;
static constexpr double gigabecquerel = 1.e+9 * becquerel;
static constexpr double millicurie    = 1.e-3 * curie;
static constexpr double microcurie    = 1.e-6 * curie;

static constexpr double Bq  = becquerel;
static constexpr double kBq = kilobecquerel;
static constexpr double MBq = megabecquerel;
static constexpr double GBq = gigabecquerel;
static constexpr double Ci  = curie;
static constexpr double mCi = millicurie;
static constexpr double uCi = microcurie;

//
// Absorbed dose [L^2][T^-2]
//
static constexpr double gray      = joule / kilogram;
static constexpr double kilogray  = 1.e+3 * gray;
static constexpr double milligray = 1.e-3 * gray;
static constexpr double microgray = 1.e-6 * gray;

//
// Luminous intensity [I]
//
static constexpr double candela = 1.;

//
// Luminous flux [I]
//
static constexpr double lumen = candela * steradian;

//
// Illuminance [I][L^-2]
//
static constexpr double lux = lumen / meter2;

//
// Miscellaneous
//
static constexpr double perCent     = 0.01;
static constexpr double perThousand = 0.001;
static constexpr double perMillion  = 0.000001;

} // namespace units
} // namespace celeritas
