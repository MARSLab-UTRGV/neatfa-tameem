package edu.unm;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * @author John Ericksen
 */
public class StatisticsRunner {

    private static final Random RAND = new Random(System.currentTimeMillis());
    private static final String MAX0_CHROMOSOME = "384.0,1,1,17,6.0061607645799935;385.0,1,2,17,5.870191371830251;386.0,1,3,17,1.5520810699168193;387.0,1,4,17,2.9947767294356704;388.0,1,5,17,-4.063858242250255;389.0,1,6,17,15.755786255018824;390.0,1,7,17,4.730416854617351;391.0,1,8,17,0.49020527554337695;392.0,1,9,17,-8.299605031895297;393.0,1,10,17,3.1874628315521716;394.0,1,11,17,-10.411009906081214;395.0,1,12,17,2.2163797720383345;396.0,1,13,17,2.306783727377028;397.0,1,14,17,-4.216312962924763;398.0,1,15,17,0.9751913346574943;399.0,1,16,17,1.7039250659925809;400.0,1,17,17,-5.660165670011577;401.0,1,18,17,-7.2934000772533;402.0,1,19,17,-0.39544310282917294;403.0,1,20,17,0.07857710953923336;405.0,1,22,17,-2.0310152626279128;406.0,1,23,17,0.010162880696185317;407.0,1,24,17,-2.643376317802906;408.0,1,1,18,0.5007110934005629;409.0,1,2,18,-2.878887159932786;410.0,1,3,18,3.5025080533028707;411.0,1,4,18,-13.44038642449706;412.0,1,5,18,-0.49515989098704094;413.0,1,6,18,12.013358963445112;414.0,1,7,18,-4.270330136746952;415.0,1,8,18,-5.033152334710261;416.0,1,9,18,-0.4394691136890628;417.0,1,10,18,2.231782260990624;418.0,1,11,18,-0.05219671355413469;419.0,1,12,18,9.78380632773108;420.0,1,13,18,-5.01020609685681;421.0,1,14,18,2.693663495030184;422.0,1,15,18,-2.144842133515894;423.0,1,16,18,-0.9920041371946777;424.0,1,17,18,1.101220739189582;425.0,1,18,18,9.603797790508345;426.0,1,19,18,1.340476112662293;427.0,1,20,18,0.09472542210071722;429.0,1,22,18,-0.6314703373850122;430.0,1,23,18,4.728170206832674;431.0,1,24,18,0.5452638792779689;432.0,1,1,19,-1.4427059384230254;433.0,1,2,19,1.1703949961499875;434.0,1,3,19,1.8756381267030435;435.0,1,4,19,-0.08184104185776842;436.0,1,5,19,1.2790795167352567;437.0,1,6,19,-7.246161711904067;438.0,1,7,19,-3.777378841537776;439.0,1,8,19,-4.594574450689551;440.0,1,9,19,-6.0677973171346355;441.0,1,10,19,1.086441981178166;442.0,1,11,19,-0.17707945971072891;443.0,1,12,19,8.499876050843955;444.0,1,13,19,0.5960649377126024;445.0,1,14,19,22.068571992234318;446.0,1,15,19,-6.902285758915365;447.0,1,16,19,-4.019998092171763;448.0,1,17,19,2.1491836260173094;449.0,1,18,19,1.427382487296816;450.0,1,19,19,0.7914870703650645;451.0,1,20,19,-1.8288823399482736;453.0,1,22,19,-2.5043474955201477;454.0,1,23,19,-1.7978155385973271;455.0,1,24,19,4.766673354088176;456.0,1,1,20,-4.198756734019529;457.0,1,2,20,3.9912245423426675;458.0,1,3,20,-5.493990945029138;459.0,1,4,20,6.6436543321104695;460.0,1,5,20,1.526826894820171;461.0,1,6,20,0.8248775819371713;462.0,1,7,20,2.983183927173001;463.0,1,8,20,3.9176069063644157;464.0,1,9,20,-2.971017898824737;465.0,1,10,20,-5.47475913705775;466.0,1,11,20,-8.351896457688346;467.0,1,12,20,-3.384305853207129;468.0,1,13,20,7.543377471657735;469.0,1,14,20,-10.748615864465354;470.0,1,15,20,-0.06264857073800867;471.0,1,16,20,0.40883182993305445;472.0,1,17,20,-1.6635710312127028;473.0,1,18,20,7.656182901195589;474.0,1,19,20,-4.537706441486187;475.0,1,20,20,-12.276878325905242;477.0,1,22,20,-5.354825340086004;478.0,1,23,20,-5.15634830113086;479.0,1,24,20,-3.4528623122421087;504.0,1,1,22,6.672419933375684;505.0,1,2,22,2.096266150653715;506.0,1,3,22,3.3091210728093854;507.0,1,4,22,-1.0107802334185374;508.0,1,5,22,-1.1412397322549142;509.0,1,6,22,-3.0068404419144246;510.0,1,7,22,9.51071691231868;511.0,1,8,22,-2.123909002383628;512.0,1,9,22,0.49230059698510464;513.0,1,10,22,-5.247585395883935;514.0,0,11,22,0.4431034275990624;515.0,1,12,22,6.312662937757593;516.0,1,13,22,2.231514085381752;517.0,1,14,22,2.6678842193105314;518.0,1,15,22,-3.6051511369269074;519.0,1,16,22,1.9951612219605952;520.0,1,17,22,1.52419828745808;521.0,1,18,22,10.13150513203471;522.0,1,19,22,-11.147778352715104;523.0,1,20,22,-3.753025576120798;525.0,1,22,22,-1.0718710186638944;526.0,1,23,22,1.4068927734153682;527.0,1,24,22,-1.7403490054792374;528.0,1,1,23,-1.4118354416927967;529.0,1,2,23,1.1221471371953164;530.0,1,3,23,4.951850195392979;531.0,1,4,23,11.382848143100743;532.0,1,5,23,2.187427865994384;533.0,1,6,23,3.0593046402003647;534.0,1,7,23,-3.402112322480309;535.0,1,8,23,-8.441451866021564;536.0,1,9,23,-12.018928942841448;537.0,1,10,23,-0.6138209192692385;538.0,1,11,23,-0.38661309117289067;539.0,1,12,23,1.9652514318132013;540.0,1,13,23,3.519359955040996;541.0,1,14,23,-2.1548790215035107;542.0,1,15,23,-1.4664876098432753;543.0,1,16,23,-1.5304140669833215;544.0,1,17,23,-0.0489285174394638;545.0,1,18,23,1.3875429792367435;546.0,1,19,23,0.368345689719761;547.0,1,20,23,9.371101861724314;549.0,1,22,23,-6.641080550710459;550.0,1,23,23,-7.240008421820672;551.0,1,24,23,0.366189853540961;552.0,1,1,24,0.4487532354603335;553.0,1,2,24,-1.6344190849662976;554.0,1,3,24,2.714294480830283;555.0,1,4,24,3.50591862029938;556.0,1,5,24,-0.5960618917757059;557.0,1,6,24,-2.6030140534131374;558.0,1,7,24,-1.938783646065855;559.0,1,8,24,0.7865655716568384;560.0,1,9,24,-0.018640081784932816;561.0,1,10,24,-0.6146489296582167;562.0,1,11,24,0.9575497161015496;563.0,1,12,24,15.385337966428573;564.0,1,13,24,3.6167742732878674;565.0,0,14,24,4.689506094163504;566.0,1,15,24,5.826842715369645;567.0,1,16,24,-3.6375126833303195;568.0,1,17,24,5.825760102964886;569.0,1,18,24,2.6658618246080015;570.0,1,19,24,-4.926907905808469;571.0,1,20,24,-3.1209617713940063;573.0,1,22,24,2.0191969966943417;574.0,1,23,24,-7.640798480133171;575.0,1,24,24,10.086061606803245;686.0,1,9,61,-1.972392316917103;687.0,1,61,19,-0.021195427871646577;694.0,1,1,61,-4.4878890869995836;729.0,1,18,61,5.107001690591936;735.0,1,6,70,2.547138149329898;736.0,1,70,24,-1.1141288829888563;753.0,1,13,75,-0.05847044878732577;754.0,1,75,20,0.7945358221422192;761.0,0,7,77,-0.4429101250339347;762.0,1,77,22,-2.5868135013589533;775.0,1,70,19,7.502210367080785;783.0,1,17,77,-1.2922106893688428;812.0,1,15,70,-11.071456810160377;862.0,1,9,75,-9.356556848902049;880.0,1,14,61,-3.0435936066516307;886.0,1,5,77,-2.3900230348293494;905.0,1,11,75,-0.5538120961459965;917.0,1,75,24,1.9966880216971006;930.0,1,11,77,-0.7422492074537179;952.0,1,70,20,1.3902187579027618;960.0,1,9,70,-1.9763613865438194;1007.0,1,12,75,10.654238224605319;1051.0,1,75,17,0.43934646126865995;1088.0,1,16,75,1.647305083753206;1096.0,1,5,75,4.049659737290959;1121.0,1,15,159,-2.34556870786358;1122.0,1,159,20,-0.5765686593237046;1151.0,1,61,23,-4.309106609872327;1185.0,1,10,159,2.1764523383848946;1212.0,1,159,70,3.5543285502919626;1290.0,1,10,70,-3.625427571047888;1299.0,1,159,23,-1.8516450978529668;1322.0,1,2,201,0.3028844383770759;1323.0,1,201,23,-0.5054764940282945;1332.0,1,8,201,-0.40810653319844126;1375.0,1,9,216,-1.7144033637201854;1376.0,1,216,20,0.1612894858615379;1391.0,1,7,70,4.657988989716993;1401.0,1,13,61,7.418267579177422;1427.0,1,3,70,4.425048695068092;1437.0,1,19,228,6.877149106263063;1438.0,1,228,22,-1.588394807165963;1475.0,1,17,228,1.309792191107551;1495.0,0,201,216,0.554995538634073;1513.0,1,2,245,-4.054406954873737;1514.0,1,245,22,-1.1348058270330832;1527.0,1,13,250,-3.101908771291659;1528.0,1,250,24,-2.8638298135061926;1552.0,1,19,245,2.296728116779232;1573.0,1,13,259,8.391810924511232;1574.0,1,259,18,1.513429611270488;1581.0,0,15,75,1.1188379191133238;1587.0,1,1,261,1.5994474867787885;1588.0,1,261,20,1.789481232176748;1609.0,1,6,159,-5.855709602280969;1612.0,1,15,267,8.980298910921661;1613.0,1,267,75,1.2288571698200998;1639.0,1,2,216,3.3686604228266024;1668.0,1,14,216,1.4280336611495574;1680.0,1,7,284,3.7022856315553776;1681.0,1,284,77,-0.15203619473838614;1774.0,1,7,304,-0.9944612620719038;1775.0,1,304,284,2.2229223073946036;1800.0,1,6,284,1.6959753245137823;1805.0,1,13,308,-1.075628700876974;1806.0,1,308,23,1.7099011611503756;1860.0,1,13,216,-0.09038083213353232;1878.0,1,267,22,-0.5424001975976551;1884.0,1,267,61,-8.897486291707779;1903.0,1,250,228,1.8563581305316923;1938.0,1,11,343,1.952803771738204;1939.0,1,343,22,-2.6540447714312076;1970.0,1,7,159,4.4694656226713345;2019.0,1,201,366,-0.37791664880621145;2020.0,1,366,216,-6.071279654009913;2042.0,1,14,370,-2.751706095297603;2043.0,1,370,24,-0.31732207691964914;2057.0,1,17,370,2.7948475069659118;2067.0,1,17,375,-2.2015679424398282;2068.0,0,375,19,-4.021691754509698;2093.0,1,18,216,4.676109274048833;2130.0,1,16,366,6.655545983695193;2155.0,1,13,370,2.252821869888194;2162.0,1,9,366,0.18360290886138442;2167.0,1,250,304,-7.721911706661943;2171.0,1,375,397,0.2783408866223952;2172.0,1,397,19,1.0063335917706553;2196.0,1,70,77,-0.7341114431326398;";
    private static final String MAX1_CHROMOSOME = "384.0,1,1,17,8.028600474772752;385.0,1,2,17,0.8803870389563675;386.0,1,3,17,-2.847182328382662;387.0,1,4,17,9.187550930544715;388.0,1,5,17,-0.7861335932469891;389.0,1,6,17,-3.845972840983915;390.0,1,7,17,1.0143735282046085;391.0,1,8,17,-2.848536358344877;392.0,1,9,17,-1.6278644686202552;393.0,1,10,17,-3.1509220969430074;394.0,1,11,17,-3.6433257127958893;395.0,1,12,17,-0.892751900967681;396.0,0,13,17,1.62847060767653;397.0,1,14,17,-2.4771641320572724;398.0,1,15,17,-3.371033036313692;399.0,1,16,17,-0.029057283541604084;400.0,1,17,17,-1.5113529219364361;401.0,1,18,17,-2.675679211904481;402.0,1,19,17,-0.4241598427245809;403.0,1,20,17,3.6019682306584437;405.0,1,22,17,-2.7308205210687726;406.0,1,23,17,0.9937029639040289;407.0,1,24,17,-1.5798619029517047;408.0,1,1,18,-4.784947531952087;409.0,1,2,18,0.25293883984558807;410.0,1,3,18,-3.220047387396313;411.0,1,4,18,-1.0705116134520822;412.0,1,5,18,-0.9618625282745018;413.0,1,6,18,-15.030141671337812;414.0,1,7,18,-5.035194832503831;415.0,1,8,18,-3.970275325520677;416.0,1,9,18,2.8095388332449813;417.0,1,10,18,0.06364949226564076;418.0,1,11,18,-0.5552684135074255;419.0,1,12,18,17.191536608345473;420.0,1,13,18,6.641562655076282;421.0,1,14,18,11.593775122782054;422.0,1,15,18,-21.975698342004954;423.0,1,16,18,-16.454889921767904;424.0,1,17,18,-0.2559618869252538;425.0,1,18,18,-2.2024686147233368;426.0,1,19,18,-0.9687220289862656;427.0,1,20,18,0.46739258855595633;429.0,1,22,18,-2.748036115825159;430.0,1,23,18,0.33842058010221776;431.0,1,24,18,-0.5083931454448166;432.0,1,1,19,4.100305209884496;433.0,1,2,19,-3.429354345433045;434.0,1,3,19,6.488768457385026;435.0,1,4,19,-0.2371962369503999;436.0,1,5,19,0.3648188724726321;437.0,1,6,19,-1.2680778503749144;438.0,1,7,19,-5.301757500827166;439.0,1,8,19,2.171978694330618;440.0,1,9,19,-11.845524787216238;441.0,1,10,19,0.8601054618448417;442.0,1,11,19,2.306657105591244;443.0,1,12,19,8.834481037608164;444.0,1,13,19,-2.873774829565749;445.0,1,14,19,4.135129757544961;446.0,1,15,19,0.6911184343133867;447.0,1,16,19,1.8973294315198381;448.0,1,17,19,-2.3834760480458375;449.0,1,18,19,5.356892389157319;450.0,1,19,19,26.609623801878488;451.0,1,20,19,4.855181934326963;453.0,1,22,19,-4.132200399687672;454.0,1,23,19,-3.7434587605589003;455.0,1,24,19,-7.600116347983508;456.0,1,1,20,1.5469351927824757;457.0,1,2,20,6.4912459737621635;458.0,1,3,20,-2.111434189745451;459.0,1,4,20,1.4254130040891932;460.0,1,5,20,1.583424978335668;461.0,1,6,20,-3.7361739944859886;462.0,1,7,20,2.3054105267429805;463.0,1,8,20,9.40026927808799;464.0,1,9,20,-5.398740110447147;465.0,1,10,20,1.6467863296957725;466.0,1,11,20,-0.7045866883611516;467.0,1,12,20,21.464972592184502;468.0,1,13,20,11.01307166352585;469.0,1,14,20,-3.455494904582583;470.0,1,15,20,-6.704510766764235;471.0,1,16,20,3.3634383325969974;472.0,1,17,20,-9.249394790869871;473.0,1,18,20,-10.496973057662874;474.0,1,19,20,-1.0599170327076635;475.0,1,20,20,0.8805105527356265;477.0,1,22,20,0.49874364075204697;478.0,1,23,20,-10.484365005415968;479.0,1,24,20,-2.3527800654963276;504.0,1,1,22,5.277068692459183;505.0,1,2,22,-0.8224092450602266;506.0,1,3,22,7.828798445662603;507.0,1,4,22,2.183702251355297;508.0,1,5,22,3.505472430053624;509.0,1,6,22,-0.4430874772598863;510.0,1,7,22,-5.6556993340715715;511.0,1,8,22,-0.7315004670520476;512.0,1,9,22,3.6820679186236496;513.0,1,10,22,-2.8460708101047945;514.0,1,11,22,-5.89759461578781;515.0,1,12,22,-3.316779245411346;516.0,1,13,22,-0.7246452223179813;517.0,1,14,22,-2.5859142900678282;518.0,1,15,22,8.447188960476156;519.0,1,16,22,3.240599868946243;520.0,1,17,22,-2.4517445430687768;521.0,1,18,22,5.199740186132561;522.0,1,19,22,2.106077271749418;523.0,1,20,22,0.816451321028736;525.0,1,22,22,-5.422469078565134;526.0,1,23,22,3.7986686681008486;527.0,1,24,22,-3.5181420655984543;528.0,1,1,23,5.922705254610397;529.0,1,2,23,-0.5340173282555852;530.0,1,3,23,-2.1796801074932795;531.0,1,4,23,-4.97416784497733;532.0,1,5,23,4.88072424967093;533.0,1,6,23,1.6529736459185378;534.0,1,7,23,-3.6129979006189283;535.0,1,8,23,1.8696720906332742;536.0,1,9,23,-4.182838718776413;537.0,1,10,23,-2.3584639716301297;538.0,1,11,23,8.420997370400357;539.0,1,12,23,-2.465775008748386;540.0,1,13,23,0.9359202099746231;541.0,1,14,23,-10.875172437261451;542.0,1,15,23,5.923157669624139;543.0,1,16,23,-0.1462174169201548;544.0,1,17,23,4.266332974094199;545.0,1,18,23,-8.326476073382448;546.0,1,19,23,-10.733336944247425;547.0,1,20,23,-0.12132560125738234;549.0,0,22,23,1.473104615937452;550.0,1,23,23,1.0290503801492523;551.0,1,24,23,-4.512937588113434;552.0,1,1,24,2.3599343187674364;553.0,1,2,24,-2.287336914853928;554.0,1,3,24,-16.77368195618871;555.0,1,4,24,3.3013367088027987;556.0,1,5,24,4.115408308344422;557.0,1,6,24,-7.877750577852021;558.0,1,7,24,-3.2233395773590674;559.0,1,8,24,3.023170384445602;560.0,1,9,24,-2.9757846503535728;561.0,1,10,24,4.539466341246331;562.0,1,11,24,-0.5228338289908165;563.0,1,12,24,0.7461117089554155;564.0,1,13,24,0.6238749064632438;565.0,1,14,24,1.4409382001816922;566.0,1,15,24,-0.2876363985213408;567.0,1,16,24,4.718730506128457;568.0,1,17,24,-3.317908884578273;569.0,1,18,24,0.9072807333232427;570.0,1,19,24,-10.95472899552938;571.0,1,20,24,-7.553375039428954;573.0,1,22,24,-15.383697131566654;574.0,1,23,24,1.6172374157262757;575.0,1,24,24,-5.118584249534485;595.0,1,11,34,-1.7676744790918804;596.0,1,34,19,-1.6565329400059636;604.0,1,1,34,-0.2737257387536206;648.0,1,12,34,1.846057237968012;687.0,1,17,34,-2.4590111533505827;705.0,1,4,34,-11.609824716132044;714.0,1,18,34,1.5153501843367791;729.0,1,5,34,4.319092024950006;754.0,1,2,34,-0.4515667769628928;821.0,1,2,93,2.4783955780956806;822.0,1,93,24,-8.574485177406041;909.0,1,24,114,-6.359406987356758;910.0,1,114,23,-5.398238435569436;934.0,1,14,114,13.400646157481695;988.0,1,16,93,10.412240116708908;1055.0,1,23,145,3.654648471073288;1056.0,1,145,24,4.789017033073346;1131.0,0,145,93,-2.331355729643694;1191.0,1,11,181,10.09575430918094;1192.0,1,181,20,1.5793042017499668;1311.0,1,5,209,-10.74169508604704;1312.0,1,209,34,-4.1619305840673535;1474.0,1,14,245,4.146131199293555;1475.0,1,245,114,4.98572084861158;1495.0,1,7,114,-3.474605419480166;1509.0,1,245,34,-3.7539965066127228;1595.0,1,245,209,1.921625242914235;1647.0,1,14,34,-4.500100777947498;1716.0,1,8,301,-0.34458417230219945;1717.0,1,301,20,9.398487328253864;1737.0,1,6,305,9.806792872632201;1738.0,1,305,22,0.4358179345163007;1759.0,1,19,308,1.5908169229872584;1760.0,1,308,23,-2.538999580963323;1783.0,1,16,34,4.881426664364609;1863.0,1,11,209,1.8901280495799544;1894.0,1,14,343,0.1957371485941406;1895.0,1,343,24,-12.474888445193667;1901.0,1,14,345,-3.7986337700984745;1902.0,1,345,343,0.18133629238873317;1927.0,1,10,209,-3.082394234147398;1937.0,1,4,145,-2.459219536803534;2064.0,1,14,378,1.514111776932317;2065.0,1,378,343,2.8438176257846264;2225.0,1,181,24,-2.4938613767060933;2243.0,1,13,421,4.564180384613456;2244.0,1,421,18,7.136228343175784;2305.0,1,308,93,-0.5573042899163305;2394.0,1,10,343,-1.9738051928901204;2412.0,1,181,463,3.664376391820804;2413.0,1,463,24,-3.108482872037288;2419.0,1,15,421,0.5339462955167048;2447.0,1,13,471,-0.09910019074805532;2448.0,1,471,19,0.8803336784597094;2459.0,1,17,475,-6.046634252054017;2460.0,1,475,22,0.096349716422519;2480.0,1,181,19,-3.585575087751611;2489.0,1,13,181,-0.6319861899256232;2561.0,0,23,343,-0.8818003587505567;2590.0,1,114,20,-2.6610323478878968;2599.0,1,11,501,0.24255554377619182;2600.0,1,501,24,-0.46555252372314426;2614.0,1,378,145,-2.480855682953371;2665.0,1,5,305,-1.310459876392149;2790.0,1,1,542,1.3395625724937517;2791.0,1,542,22,0.09361447532752032;2887.0,1,471,181,0.4092105571016007;2900.0,1,7,181,-2.787761363904371;2968.0,1,11,378,-1.4428803536026087;2985.0,1,181,345,-1.3754569152422635;3010.0,1,345,475,1.0252668694402365;3086.0,1,8,601,-0.8944403630902111;3087.0,1,601,18,4.713651805085513;3091.0,1,114,463,-0.914503630255654;3155.0,1,8,34,-0.2651224652069242;3170.0,1,1,616,4.832035946740513;3171.0,1,616,20,1.9625071345311236;3191.0,1,18,378,-1.1973072704676535;3335.0,1,13,601,1.5370163880069265;3361.0,1,9,660,2.574490275274325;3362.0,1,660,18,-0.40201767069378413;3402.0,1,11,671,1.6033752623132491;3403.0,1,671,20,-0.07113985221272401;3444.0,1,3,542,0.6987586326600643;3482.0,1,20,698,-0.3922113370499515;3483.0,1,698,18,-0.6148257065517702;3494.0,1,181,475,1.8225943587957478;3513.0,1,4,308,1.1408636710406463;3638.0,1,13,731,0.5310280369574363;3639.0,1,731,19,0.8446643527926275;3664.0,1,181,301,-4.826662545303729;3717.0,1,9,463,-2.088645218365;3736.0,1,181,114,-1.3497577178078688;3745.0,1,671,308,-3.3400937124685157;3835.0,1,19,775,1.92450476203772;3836.0,1,775,17,0.6039411480764656;3850.0,1,475,501,-0.5526037422498263;3870.0,1,660,93,7.152223368449824;3887.0,1,23,779,-0.6183349645016636;3888.0,1,779,343,-5.338156706413843;3955.0,1,145,793,1.0;3956.0,1,793,93,-0.6900350618645079;3962.0,1,209,501,5.621604528138184;3979.0,1,22,797,1.0;3980.0,1,797,23,1.473104615937452;4000.0,1,660,34,-5.642398177909969;4004.0,1,20,463,-2.229720045334801;4018.0,1,181,698,2.448146332607962;4032.0,1,13,810,1.0;4033.0,0,810,17,1.62847060767653;4049.0,1,13,616,-8.950936609922993;4061.0,1,810,815,1.0;4062.0,1,815,17,1.62847060767653;";
    private static final String MAX2_CHROMOSOME = "384.0,1,1,17,-1.5601071303445617;385.0,1,2,17,2.59205524239083;386.0,1,3,17,4.415745683757838;387.0,1,4,17,-1.5250193791825841;388.0,1,5,17,1.9088604796418847;389.0,1,6,17,14.57757321760522;390.0,1,7,17,3.1675975190785155;391.0,1,8,17,-1.1465203804283957;392.0,1,9,17,-8.261697310246959;393.0,1,10,17,2.7063950590025745;394.0,1,11,17,-3.1304160005282515;395.0,1,12,17,-1.5021954238197925;396.0,1,13,17,0.14732991948297414;397.0,1,14,17,-2.669555034333055;398.0,1,15,17,-3.396226475693126;399.0,1,16,17,-13.347060688398232;400.0,1,17,17,-12.217018204588843;401.0,1,18,17,-5.489018022962014;402.0,1,19,17,5.8260878212369835;403.0,1,20,17,-1.1365459399152225;405.0,1,22,17,5.858386001307255;406.0,1,23,17,-11.607921933346708;407.0,1,24,17,0.43154495043401114;408.0,1,1,18,1.246484565963086;409.0,1,2,18,-0.6296094659543969;410.0,1,3,18,-3.9319496528662046;411.0,1,4,18,-4.997507806105776;412.0,1,5,18,-1.603763797543327;413.0,1,6,18,22.622229211573526;414.0,1,7,18,-6.303102168005178;415.0,1,8,18,-8.808934583421491;416.0,1,9,18,-5.995469340438794;417.0,1,10,18,7.56770009959611;418.0,1,11,18,5.278941201990631;419.0,1,12,18,-26.77435918072439;420.0,1,13,18,21.02605706794256;421.0,1,14,18,3.3838958566488317;422.0,1,15,18,-2.447035232472501;423.0,1,16,18,-18.700108696789826;424.0,1,17,18,0.7285038060375262;425.0,1,18,18,-6.314303157872001;426.0,1,19,18,4.883757069977669;427.0,1,20,18,1.9228081005711595;429.0,1,22,18,-7.531464112583295;430.0,1,23,18,2.6164965580901183;431.0,1,24,18,0.34135410449215975;432.0,1,1,19,1.2282819547887067;433.0,1,2,19,-5.7256308840479075;434.0,1,3,19,2.536570453809695;435.0,1,4,19,1.7719682131738677;436.0,1,5,19,1.4855750675983714;437.0,1,6,19,3.7718157560460446;438.0,1,7,19,-0.9132076298345615;439.0,1,8,19,9.596907015606762;440.0,1,9,19,-0.7158011783825446;441.0,1,10,19,-2.6841720288063113;442.0,1,11,19,-0.4446696438880202;443.0,1,12,19,-22.290917400189567;444.0,1,13,19,-11.372578977051466;445.0,1,14,19,-0.5691996273405674;446.0,1,15,19,-0.32945514761770145;447.0,1,16,19,-3.1686377991383803;448.0,1,17,19,4.930033750028928;449.0,1,18,19,-0.1922045159760653;450.0,1,19,19,7.979561382958828;451.0,1,20,19,-2.0580447686016106;453.0,1,22,19,-10.922827891133476;454.0,1,23,19,9.134627255720416;455.0,1,24,19,2.1901642390153415;456.0,1,1,20,1.9778278229491026;457.0,1,2,20,0.3275523035735417;458.0,1,3,20,-4.303317470529545;459.0,1,4,20,-15.926050520116537;460.0,1,5,20,8.515333724435362;461.0,1,6,20,5.472352303605213;462.0,1,7,20,-13.749017923584415;463.0,1,8,20,0.007938470169496004;464.0,1,9,20,4.881304066729827;465.0,1,10,20,3.937544029797594;466.0,1,11,20,8.808597747830563;467.0,1,12,20,1.6799916134351744;468.0,1,13,20,-2.599462307020981;469.0,1,14,20,-0.5711583139613994;470.0,1,15,20,0.1270576455571093;471.0,1,16,20,-1.0131287328253538;472.0,1,17,20,4.922506832754975;473.0,1,18,20,2.3874822349748492;474.0,1,19,20,-12.13559479396946;475.0,1,20,20,-4.8344401174528535;477.0,1,22,20,6.355102365435763;478.0,1,23,20,8.678833669697722;479.0,1,24,20,-5.208317371838069;504.0,1,1,22,6.8478828201892235;505.0,1,2,22,-3.0089802983414478;506.0,1,3,22,-6.187698040124477;507.0,1,4,22,2.1677284985927856;508.0,1,5,22,-0.10724719091378809;509.0,1,6,22,-9.480941802818187;510.0,1,7,22,-4.087653852970692;511.0,1,8,22,3.785901848924509;512.0,1,9,22,4.812751648594959;513.0,1,10,22,17.232976759954994;514.0,1,11,22,0.06379540010931328;515.0,1,12,22,5.329606753633168;516.0,1,13,22,-1.553567621342824;517.0,1,14,22,-10.894876588492846;518.0,1,15,22,-7.485318223397618;519.0,1,16,22,3.0117899501779974;520.0,1,17,22,14.125541995199885;521.0,1,18,22,-3.2431172389766685;522.0,1,19,22,0.5754507248799767;523.0,1,20,22,-11.99956089965792;525.0,1,22,22,-14.7637193376869;526.0,1,23,22,6.011597399951869;527.0,1,24,22,7.426336967031224;528.0,1,1,23,0.8025045987264212;529.0,1,2,23,5.147343618380583;530.0,1,3,23,-12.925307524280257;531.0,1,4,23,-1.207164967681042;532.0,1,5,23,0.0885129129603126;533.0,1,6,23,-6.904761420727107;534.0,1,7,23,3.76562408050498;535.0,1,8,23,0.12567807853087576;536.0,1,9,23,4.457672738805531;537.0,1,10,23,-24.548996237434185;538.0,1,11,23,15.13312841459687;539.0,1,12,23,1.055045713263521;540.0,1,13,23,4.668459894196594;541.0,1,14,23,-4.533797066748146;542.0,1,15,23,-5.886260852213871;543.0,1,16,23,0.23891249868755393;544.0,1,17,23,-7.705624138429187;545.0,1,18,23,5.472177320287566;546.0,1,19,23,-3.2073799240025176;547.0,1,20,23,0.8861006660371209;549.0,1,22,23,7.907630402922411;550.0,1,23,23,-2.3749937392254115;551.0,1,24,23,-0.14933668666907451;552.0,1,1,24,-5.6527961168183065;553.0,1,2,24,-2.9707414499423104;554.0,1,3,24,-7.225208942016317;555.0,1,4,24,1.7868492436409609;556.0,1,5,24,-5.922106748488277;557.0,1,6,24,3.2879071979422574;558.0,1,7,24,8.561555501889067;559.0,1,8,24,12.569226414103001;560.0,1,9,24,-13.395293122854135;561.0,1,10,24,9.652788368450306;562.0,1,11,24,-2.502974402472154;563.0,1,12,24,-23.233288014503405;564.0,1,13,24,16.165175002192736;565.0,1,14,24,5.0428130196066805;566.0,1,15,24,0.9333440671535439;567.0,1,16,24,-3.675584320357034;568.0,1,17,24,6.551052179540637;569.0,1,18,24,2.5890571176241766;570.0,1,19,24,-1.4426653365705582;571.0,1,20,24,12.02777702348299;573.0,1,22,24,0.21634034259094825;574.0,1,23,24,-6.781839076073919;575.0,1,24,24,5.963089248255773;805.0,1,1,97,-0.36274828517278324;806.0,1,97,18,-21.32749004105444;815.0,1,8,97,-6.290842163237082;895.0,1,18,118,-2.6721755807315817;896.0,1,118,18,-1.0523898786945518;911.0,1,3,97,-10.537332661652048;930.0,1,11,128,-3.062053127057028;931.0,1,128,20,-2.084997155923449;953.0,1,11,97,-13.009379039651995;979.0,1,14,118,-4.365682517174396;1004.0,1,17,97,-0.6768435032560007;1041.0,1,10,118,4.1445631868277415;1087.0,1,16,97,14.141435179789573;1090.0,1,3,128,0.14311920844094989;1119.0,1,6,172,8.631526374487308;1120.0,1,172,19,4.580959179430296;1169.0,1,118,128,-5.294912347860155;1196.0,1,3,187,1.6958616948467111;1197.0,1,187,128,9.333137697957675;1252.0,1,8,202,-0.2472124776064113;1253.0,1,202,17,-3.7682255625956156;1489.0,1,10,257,4.063438401197603;1490.0,1,257,23,-0.6833498851694939;1526.0,1,257,22,-1.4803639417293928;1534.0,1,4,266,-4.45716188387926;1535.0,1,266,20,-5.79416906448491;1593.0,1,16,187,0.8209376676069162;1682.0,1,2,303,1.9165266188157717;1683.0,1,303,24,0.22189271651882492;1758.0,1,17,322,0.48053903530561937;1759.0,1,322,20,-1.4848184372766675;1844.0,1,7,97,0.18170926247508315;1892.0,1,187,23,1.4329480968276533;1947.0,1,15,257,1.985405501078935;1958.0,1,14,257,0.5005549619682875;1966.0,1,12,172,0.7613314618043288;2017.0,1,2,376,0.6232550213091147;2018.0,1,376,18,1.3608006544119484;2037.0,1,10,322,0.3030274165210445;2064.0,1,1,128,-1.5032572718319233;2076.0,1,11,266,-1.3376115165566795;2096.0,1,6,376,2.5298370188622106;2161.0,1,13,172,1.01188298805934;2231.0,1,18,419,-0.5425590832030649;2232.0,1,419,24,-3.180056086917622;2242.0,1,257,322,3.5188643637449446;2260.0,1,266,18,0.5211176495158049;2380.0,1,22,447,-1.443663568421565;2381.0,1,447,20,1.6455371838410673;2446.0,1,419,20,1.802713673461308;2467.0,1,3,376,-4.28986545347402;2480.0,1,17,257,1.2435869625752538;2641.0,1,10,507,0.13561757750721723;2642.0,1,507,24,-1.1489105915372366;2658.0,1,16,172,1.2729938052412417;2711.0,1,17,522,-0.9627678502604526;2712.0,1,522,22,-1.3757934045823883;2727.0,1,376,172,0.8188254910342321;2769.0,1,7,172,5.972435656488881;2827.0,1,15,507,-1.4979106378413576;2831.0,1,18,447,-0.8446251349998981;2846.0,1,11,551,0.15933214165886467;2847.0,1,551,23,-0.4011331414548207;2857.0,1,18,522,1.026592387850148;2900.0,1,172,551,1.15734003940495;2928.0,1,118,376,0.5835546903194719;2938.0,1,15,564,-2.2441271943712504;2939.0,1,564,507,-0.3805196912079445;3041.0,1,8,322,-6.196571342777095;3108.0,1,9,322,2.493637239113112;3278.0,1,2,522,-3.1164534570958002;3538.0,1,19,187,-1.6650147606260193;3614.0,1,97,564,1.495794807138883;3662.0,1,13,507,1.4267503446312615;3739.0,1,97,376,-2.462690705253288;3775.0,1,2,752,1.0;3776.0,1,752,303,-2.081252519125925;";
    private static final String MAX_GENERNAL_CHROMOSOME = "384.0,1,1,17,5.408086065462844;385.0,1,2,17,-0.23190643663276544;386.0,1,3,17,-2.3461577653334924;387.0,1,4,17,-13.958841860400122;388.0,1,5,17,3.732994418297022;389.0,1,6,17,1.4730212478296711;390.0,1,7,17,0.5274268070972256;391.0,1,8,17,3.0639764271503243;392.0,1,9,17,-10.002406689584008;393.0,1,10,17,-8.279603494265817;394.0,1,11,17,-1.7455882814424077;395.0,1,12,17,-2.9043421594792425;396.0,1,13,17,1.8628561500347396;397.0,1,14,17,8.150425935140904;398.0,1,15,17,-2.6652199159046726;399.0,1,16,17,-6.113128375364086;400.0,1,17,17,-9.67708838258092;401.0,1,18,17,-1.7412375137782605;402.0,1,19,17,-0.507223948414501;403.0,1,20,17,6.90346806531063;405.0,1,22,17,3.0350413168131825;406.0,1,23,17,-0.7810115513298841;407.0,1,24,17,-2.0911751635852815;408.0,1,1,18,0.07316326885394009;409.0,1,2,18,0.10973454122092763;410.0,1,3,18,-9.820222224364905;411.0,1,4,18,9.281503812406344;412.0,1,5,18,-0.4489218993245102;413.0,1,6,18,-11.853941218483074;414.0,1,7,18,4.804632185697841;415.0,0,8,18,-3.102521085480835;416.0,1,9,18,-1.8759532658400735;417.0,1,10,18,-2.3276479898264535;418.0,1,11,18,-5.929113722640304;419.0,1,12,18,24.259429425377775;420.0,1,13,18,16.39109782247685;421.0,1,14,18,-6.743437411204477;422.0,1,15,18,-4.871275729823189;423.0,1,16,18,-8.127065691535645;424.0,1,17,18,1.1301687945803838;425.0,1,18,18,-8.007545299874922;426.0,1,19,18,-16.275442885402867;427.0,1,20,18,1.214768124141492;429.0,1,22,18,1.0178671905816812;430.0,1,23,18,1.050293739883141;431.0,1,24,18,6.389692267909458;432.0,0,1,19,0.6697823896179169;433.0,1,2,19,-3.8575740326397483;434.0,1,3,19,7.487594989568634;435.0,0,4,19,-3.2333078660260304;436.0,1,5,19,-0.7915865388103116;437.0,1,6,19,-2.3025889227217258;438.0,1,7,19,-4.560877748680277;439.0,1,8,19,-5.866160775290129;440.0,1,9,19,-7.916408786393301;441.0,1,10,19,7.220559759997889;442.0,1,11,19,4.369501809407114;443.0,1,12,19,11.497081546922049;444.0,1,13,19,-5.3637284442464175;445.0,1,14,19,-3.641542147194172;446.0,1,15,19,3.82242094123718;447.0,1,16,19,-4.289202676410371;448.0,1,17,19,-6.509186957610752;449.0,1,18,19,7.7340093623372965;450.0,1,19,19,-6.134285683915426;451.0,1,20,19,6.226271389559435;453.0,1,22,19,-11.818462200178585;454.0,1,23,19,-5.392729130539489;455.0,1,24,19,11.833900345999218;456.0,1,1,20,-3.8731471559314645;457.0,0,2,20,0.7082410684475002;458.0,1,3,20,-0.5717481230443551;459.0,1,4,20,-14.005161932214877;460.0,1,5,20,12.597302992307261;461.0,1,6,20,2.581972938434969;462.0,1,7,20,8.288177971038653;463.0,1,8,20,4.423938946817206;464.0,1,9,20,10.572203359906023;465.0,1,10,20,-3.5825246734119336;466.0,1,11,20,1.1953040973525502;467.0,0,12,20,-5.945558381250088;468.0,1,13,20,-0.19483061818027325;469.0,1,14,20,8.190985998316329;470.0,1,15,20,-1.5063225700231517;471.0,1,16,20,19.995903102391;472.0,1,17,20,1.2217985219992356;473.0,1,18,20,-0.12964853991152014;474.0,1,19,20,8.566690467713594;475.0,1,20,20,-0.33425276299754625;477.0,1,22,20,2.7659707398070443;478.0,1,23,20,-6.902463080445472;479.0,1,24,20,3.155605015423227;504.0,1,1,22,3.8604505194522556;505.0,1,2,22,-4.673198437654973;506.0,1,3,22,-1.4618308371037256;507.0,1,4,22,4.30811953500235;508.0,1,5,22,-10.352886755827308;509.0,1,6,22,0.9394615377099005;510.0,1,7,22,0.2452432526954107;511.0,1,8,22,4.732276117582965;512.0,1,9,22,7.2595532659518;513.0,1,10,22,3.534943932923883;514.0,1,11,22,5.450471383144793;515.0,1,12,22,-0.5249313793675721;516.0,1,13,22,0.45864895767656777;517.0,1,14,22,2.296850483225434;518.0,1,15,22,2.9304014403777967;519.0,1,16,22,-0.17607126276130458;520.0,1,17,22,0.7852517171803026;521.0,1,18,22,5.409879398654644;522.0,1,19,22,18.974701500692017;523.0,1,20,22,0.1788669944575083;525.0,1,22,22,-1.6409565862475648;526.0,1,23,22,-9.741464275441274;527.0,1,24,22,-7.638075743094436;528.0,1,1,23,-2.2196551585224795;529.0,1,2,23,1.1414825784145273;530.0,1,3,23,1.9087158726795828;531.0,1,4,23,-4.713984290509796;532.0,1,5,23,1.3917459962703034;533.0,1,6,23,7.292067976552204;534.0,1,7,23,-5.681847240659569;535.0,1,8,23,-0.05746605846496666;536.0,1,9,23,8.676235826723374;537.0,1,10,23,-4.4891820127187305;538.0,1,11,23,-8.23462618000152;539.0,1,12,23,-4.59114857944625;540.0,1,13,23,8.955274332793987;541.0,1,14,23,-10.900667562119706;542.0,1,15,23,3.825939692912092;543.0,1,16,23,-0.51795455144355;544.0,1,17,23,-4.0201041018882595;545.0,1,18,23,-38.714229125092;546.0,1,19,23,-1.2566618537140095;547.0,1,20,23,-17.771497185400808;549.0,1,22,23,-3.5756557027991196;550.0,1,23,23,-1.924084540961977;551.0,1,24,23,0.7611911280883974;552.0,1,1,24,5.521773617357831;553.0,1,2,24,2.277985024380027;554.0,1,3,24,20.254356561414298;555.0,1,4,24,0.820675685675274;556.0,1,5,24,-5.494457618246929;557.0,1,6,24,-4.735492374883241;558.0,1,7,24,4.7881108830898365;559.0,1,8,24,2.698209672332374;560.0,1,9,24,-1.2954979936432587;561.0,1,10,24,8.358230337201626;562.0,1,11,24,-0.10533793555647364;563.0,1,12,24,14.232363708425758;564.0,1,13,24,6.2467919436310835;565.0,1,14,24,7.835114534584454;566.0,1,15,24,5.405300809254639;567.0,1,16,24,2.2679288074771953;568.0,1,17,24,-4.179788783323576;569.0,1,18,24,-1.9012414105704512;570.0,1,19,24,11.252878264150343;571.0,1,20,24,4.091479900468887;573.0,1,22,24,0.7195442036905175;574.0,1,23,24,-3.854901090702162;575.0,1,24,24,2.8009306938437244;593.0,1,19,33,-2.3854322331486126;594.0,1,33,17,3.512161530537411;600.0,1,33,24,0.5556463861192042;615.0,1,33,18,7.648395745533115;667.0,1,13,33,3.3762059436209633;676.0,1,19,54,9.496041672240878;677.0,1,54,33,3.6559371013961757;692.0,1,16,33,10.296775728107182;695.0,1,22,59,-2.7229513196204493;696.0,1,59,17,-0.4977733376567368;713.0,1,8,33,3.5506467907663724;733.0,1,15,54,-3.2486253137559915;771.0,1,5,77,-3.3543727004162793;772.0,1,77,23,1.3644576613225494;784.0,1,4,77,-0.9223419722917496;858.0,1,7,33,6.551075926949911;975.0,1,33,20,1.532384266461934;1049.0,1,8,138,-3.7899990653458357;1050.0,1,138,22,-1.4201771045391811;1058.0,1,12,59,-1.67600541449019;1178.0,1,54,17,-3.0567243434153206;1185.0,1,6,167,7.801784849050366;1186.0,1,167,22,2.0859472271570314;1196.0,1,9,170,-7.100928456067235;1197.0,1,170,18,-12.772722334331023;1236.0,1,19,178,1.4786736138518497;1237.0,1,178,24,8.496273406597167;1250.0,1,178,23,-11.505813748689997;1270.0,1,5,183,3.987719067374754;1271.0,1,183,20,-11.42061062135507;1362.0,1,13,170,-5.160362842641076;1393.0,1,24,212,2.4775486322399503;1394.0,1,212,20,9.49350542134893;1490.0,1,4,230,-2.7165556813567218;1491.0,1,230,19,5.829851204532863;1557.0,1,20,242,2.8875591559957448;1558.0,0,242,23,-0.0709033262124028;1602.0,1,183,212,-0.03357394814347758;1617.0,1,15,230,7.2442197030464985;1682.0,1,11,268,-0.5499083271188145;1683.0,1,268,20,-3.7677972749090216;1702.0,1,6,138,-2.3447753951157075;1778.0,1,8,170,2.4083992017760103;1838.0,1,13,303,3.753218691839978;1839.0,1,303,20,4.915151079968249;1857.0,1,16,59,-2.7044025748387144;1870.0,1,8,54,-0.8064936532207262;1883.0,1,16,212,-0.6966712695096764;1944.0,1,11,138,0.8246610552303119;2040.0,1,178,22,-2.994159978312173;2183.0,1,33,242,-1.0554284407740928;2190.0,1,170,178,-1.7059347632594288;2210.0,1,170,230,11.077059808833964;2344.0,1,17,183,0.17475213142496754;2410.0,1,54,167,1.5443278594110275;2416.0,1,24,435,3.3656317524945853;2417.0,1,435,17,-1.9626268875889028;2444.0,1,6,54,-1.7111097003859572;2622.0,1,4,183,-1.4823550119313431;2786.0,1,167,230,-1.1413286848829642;2825.0,1,1,524,0.3760723338424219;2826.0,1,524,24,0.20257693969783208;3013.0,1,435,24,-0.10366076802895785;3153.0,1,4,602,0.26858086123614955;3154.0,1,602,18,2.746547975775405;3317.0,1,5,178,-1.618429480990287;3389.0,1,15,602,0.3922028139723197;3412.0,1,9,659,0.02022178269345254;3413.0,1,659,17,0.7263934105381145;3442.0,1,17,602,3.053627018481195;3603.0,1,3,230,-1.6162080868147286;3616.0,0,14,698,-1.373239796652175;3617.0,1,698,24,5.860212968816783;3626.0,1,22,700,-0.2617725454426263;3627.0,1,700,17,-2.8607555640028197;3753.0,1,2,303,-2.7471061733197875;3788.0,1,54,435,-2.284966226588088;3844.0,1,435,77,-1.8892518479988132;3923.0,1,12,766,-2.875866071721279;3924.0,1,766,20,-0.5541181092472487;3937.0,1,2,768,-2.0241266430767126;3938.0,1,768,20,1.4092344199485636;3947.0,1,242,770,3.2352046388044267;3948.0,1,770,23,1.0297051583141994;3972.0,1,1,778,-4.725343655052177;3973.0,1,778,19,-0.7983531924888458;4047.0,1,698,242,-0.8497077985327068;4067.0,1,698,797,-2.5630607562065078;4068.0,1,797,24,0.23112480292145587;4138.0,1,268,183,-1.1076464207182337;4171.0,1,59,698,-4.904720321305976;4186.0,1,6,768,0.6641884873692705;4209.0,1,602,242,-0.5496307956581541;4240.0,1,18,242,6.060988201962186;4305.0,1,14,850,1.0;4306.0,1,850,698,-1.373239796652175;4312.0,1,8,852,1.0;4313.0,1,852,18,-3.102521085480835;";
    private static final String[] CHROMOSOMES = {MAX0_CHROMOSOME, MAX1_CHROMOSOME, MAX2_CHROMOSOME, MAX_GENERNAL_CHROMOSOME};
    private static final String HOME_DIRECTORY = "/home/john/dev/iAnt-ARGoS/";
    private static final String EXECUTABLE = "./build/source/iant_main";

    public static void main(String[] args) throws Exception {
        new StatisticsRunner().run();
    }

    public void run() throws Exception  {
        IAntXMLBuilder builder = new IAntXMLBuilder("iAnt.xml");

        ExperimentParameters.Builder parametersBuilder = ExperimentParameters.builder();

        parametersBuilder
                .populationSize(100)
                .runtime(30 * 60)
                .entityCount(6);

        for(int i = 0; i < 4; i++) {
            long startTime = System.currentTimeMillis();
            String chromosome = CHROMOSOMES[i];

            for(int d = 0; d < 3; d++) {

                parametersBuilder
                        .startTime(System.currentTimeMillis());


                final List<Long> fitness = new ArrayList<Long>();
                ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

                for(int j = 0; j < 40; j++) {

                    InputStream xml = new ByteArrayInputStream(builder.buildXML(chromosome, RAND.nextInt(65536), parametersBuilder.build(), d).getBytes());

                    final String tag = startTime + "F" + i + "D" + d + "C" + j;

                    executor.submit(
                            new ProcessExecutable(HOME_DIRECTORY, new String[]{EXECUTABLE, tag}, xml,
                                    new ProcessExecutable.OnResultCallback() {
                                        @Override
                                        public void onResult(Long result) {
                                            //System.out.println("Tag: " + tag + " Fitness: " + result);
                                            fitness.add(result);
                                        }
                                    }
                            ));
                }

                executor.shutdown();

                while (!executor.awaitTermination(1, TimeUnit.SECONDS)) {}
                System.out.println("For: " + i + " Dist: " + d + " Stats: " + getStatistics(fitness));
            }

        }
    }

    private static String getStatistics(List<Long> fitness) {
        long sum = 0;

        for(Long f : fitness) {
            sum += f;
        }

        double average = (sum * 1.0) / fitness.size();

        double stddevsum = 0;
        for(Long f : fitness) {
            stddevsum += Math.pow((f - average), 2);
        }

        double stddev = Math.sqrt(stddevsum / fitness.size());

        return "Average: " + (average * 100 / (3 * 256)) + " Std Dev: " + (stddev * 100 / (3 * 256));
    }
}
