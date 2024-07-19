import numpy as np

def griewank(x):
    ps, D = x.shape
    xs = x ** 2
    sum_term = np.sum(xs, axis=1) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, D+1))), axis=1)
    f = sum_term - prod_term + 1
    return f

# layers代表的是每一层的粒子数组
# d代表
# lb, ub代表下边界和上边界
# MAX_FES代表的是达到最高适应值的多少结束，相当于是迭代次数的功能
# funcid代表测试函数id
def ppso(layers, d, lb, ub, MAX_FES, funcid):
    # 划分的层的数量
    layersno = len(layers)
    # 累计的层数，粒子的累计和
    layercumsum = np.cumsum(layers)
    # 总的粒子数目
    sz = np.sum(layers)
    FES = 0
    fitness = 1e200

    if d == 50:
        phi = 0.04
    elif d == 30:
        phi = 0.02
    else:
        phi = 0.008

    # 粒子位置
    lvlp = np.zeros((layersno, max(layers), d))
    # 粒子速度
    lvlv = np.zeros((layersno, max(layers), d))
    fitnesslayer = np.zeros((layersno, max(layers), 1))
    pblayer = np.zeros((layersno, max(layers), d))
    
    # 粒子的坐标下界
    XRRmin = np.tile(lb, (sz, 1))
    # 粒子的坐标上界
    XRRmax = np.tile(ub, (sz, 1))
    # 生成粒子的坐标点
    p = XRRmin + (XRRmax - XRRmin) * np.random.rand(sz, d)
    # 生成粒子的速度
    v = 0.1 * (XRRmin + (XRRmax - XRRmin) * np.random.rand(sz, d))

    pb = p.copy()
    gbf = 1e200

    if funcid == 1:
        af = griewank(p)
    else:
        raise ValueError("Unsupported funcid")

    FES = sz
    bestever = gbf
    fitness = af
    MAXGEN = MAX_FES // sz
    fbest_hist = np.zeros(MAXGEN)

    gen = 1
    tp = np.zeros((sz, d))
    tv = np.zeros((sz, d))
    tpb = np.zeros((sz, d))
    tf = 9999999 * np.ones(sz)
    layeridxs = np.zeros((layersno, 2))
    allpos = np.zeros((sz, d))
    allv = np.zeros((sz, d))

    while FES < MAX_FES:
        fv = np.sort(fitness)
        pid = np.argsort(fitness)

        fbest_hist[gen - 1] = fv[0]
        
        idx = pid[:layercumsum[0]]
        lvlp[0, :len(idx), :] = p[idx, :]
        lvlv[0, :len(idx), :] = v[idx, :]
        fitnesslayer[0, :len(idx), :] = fitness[idx, np.newaxis]
        pblayer[0, :len(idx), :] = pb[idx, :]

        for li in range(1, layersno):
            idx = pid[layercumsum[li - 1]:layercumsum[li]]
            len_idx = len(idx)
            lvlp[li, :len_idx, :] = p[idx, :]
            lvlv[li, :len_idx, :] = v[idx, :]
            fitnesslayer[li, :len_idx, :] = fitness[idx, np.newaxis]
            pblayer[li, :len_idx, :] = pb[idx, :]

        ttidx = 0
        llosers = []
        wwiners = []
        
        for li in range(layersno - 1, -1, -1):
            lvlsize = layers[li]
            rlist = np.random.permutation(lvlsize)
            seprator = lvlsize // 2
            rpairs = np.column_stack((rlist[:seprator], rlist[seprator:2 * seprator]))
            
            mask = (fitnesslayer[li, rpairs[:, 0], 0] > fitnesslayer[li, rpairs[:, 1], 0])
            losers = np.where(mask, rpairs[:, 0], rpairs[:, 1])
            winners = np.where(~mask, rpairs[:, 0], rpairs[:, 1])

            randco1 = np.random.rand(seprator, d)
            randco2 = np.random.rand(seprator, d)
            randco3 = np.random.rand(seprator, d)
            
            lvlvlosert = lvlv[li, losers, :].reshape(seprator, d)
            lvlplosert = lvlp[li, losers, :].reshape(seprator, d)
            lvlpblosert = pblayer[li, losers, :].reshape(seprator, d)
            
            lvlvwinert = lvlv[li, winners, :].reshape(seprator, d)
            lvlpwinert = lvlp[li, winners, :].reshape(seprator, d)
            lvlpbwinert = pblayer[li, winners, :].reshape(seprator, d)
            
            toplvlsize = layers[0]
            indciestop = np.random.permutation(seprator) % toplvlsize
            gbpmat = lvlp[0, indciestop, :].reshape(seprator, d)
            
            lvlvlosert2 = randco1 * lvlvlosert + randco2 * (lvlpwinert - lvlplosert) + randco3 * (lvlpblosert - lvlplosert)
            lvlplosert2 = lvlplosert + lvlvlosert2

            if li != 0:
                upperlvlsize = layers[li - 1]
                indcies = np.random.permutation(seprator) % upperlvlsize
                upperpmat = lvlp[li - 1, indcies, :].reshape(seprator, d)

                randco1 = np.random.rand(seprator, d)
                randco2 = np.random.rand(seprator, d)
                randco3 = np.random.rand(seprator, d)
                randco4 = np.random.rand(seprator, d)

                lvlvwinert2 = randco1 * lvlvwinert + randco2 * (upperpmat - lvlpwinert) + randco3 * (lvlpbwinert - lvlpwinert) + phi * randco4 * (gbpmat - lvlpwinert)
                lvlpwinert2 = lvlpwinert + lvlvwinert2
            else:
                lvlvwinert2 = lvlvwinert
                lvlpwinert2 = lvlpwinert

            mergedlvlp = np.vstack((lvlplosert2, lvlpwinert2))
            ts1, ts2 = len(lvlplosert2), len(lvlpwinert2)
            layeridxs[li, :] = [ts1, ts2]

            mergedlvlv = np.vstack((lvlvlosert2, lvlvwinert2))
            allpos[ttidx:ttidx + ts1 + ts2, :] = mergedlvlp
            allv[ttidx:ttidx + ts1 + ts2, :] = mergedlvlv
            ttidx += ts1 + ts2
            llosers.append(losers)
            wwiners.append(winners)

        allpos = np.clip(allpos, lb, ub)
        if funcid == 1:
            ffs = griewank(allpos)
        else:
            raise ValueError("Unsupported funcid")

        ttidx = 0
        for li in range(layersno - 1, -1, -1):
            ts1, ts2 = layeridxs[li]
            ff1 = ffs[ttidx:ttidx + ts1]
            lvlplosert2 = allpos[ttidx:ttidx + ts1]
            lvlvlosert2 = allv[ttidx:ttidx + ts1]
            ff2 = ffs[ttidx + ts1:ttidx + ts1 + ts2]
            lvlpwinert2 = allpos[ttidx + ts1:ttidx + ts1 + ts2]
            lvlvwinert2 = allv[ttidx + ts1:ttidx + ts1 + ts2]
            ttidx += ts1 + ts2

            losers = llosers[li]
            winners = wwiners[li]

            goodloseridx = fitnesslayer[li, losers, 0] > ff1
            goodwinderidx = fitnesslayer[li, winners, 0] > ff2
            lidxs = losers[goodloseridx]
            widxs = winners[goodwinderidx]

            pblayer[li, lidxs, :] = lvlplosert2[goodloseridx]
            pblayer[li, widxs, :] = lvlpwinert2[goodwinderidx]

            lvlv[li, losers, :] = lvlvlosert2
            lvlp[li, losers, :] = lvlplosert2

            lvlv[li, winners, :] = lvlvwinert2
            lvlp[li, winners, :] = lvlpwinert2

            fitnesslayer[li, losers, 0] = ff1
            fitnesslayer[li, winners, 0] = ff2

        for li in range(layersno):
            if li == 0:
                leftb = 0
                rightb = layercumsum[li]
            else:
                leftb = layercumsum[li - 1]
                rightb = layercumsum[li]

            tp[leftb:rightb, :] = lvlp[li, :layers[li]].reshape(layers[li], d)
            tv[leftb:rightb, :] = lvlv[li, :layers[li]].reshape(layers[li], d)
            tpb[leftb:rightb, :] = pblayer[li, :layers[li]].reshape(layers[li], d)
            tf[leftb:rightb] = fitnesslayer[li, :layers[li], 0]

        p = tp
        v = tv
        pb = tpb

        fitness = tf
        bestever = min(bestever, fitness.min())
        FES += sz
        gen += 1

    return bestever, fitness, p, fbest_hist


